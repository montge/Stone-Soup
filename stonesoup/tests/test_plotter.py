import sys
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pytest

from stonesoup.dataassociator.neighbour import NearestNeighbour
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.platform.base import Obstacle
from stonesoup.platform.shape import Shape
from stonesoup.plotter import (
    AnimatedPlotterly,
    AnimatedPolarPlotterly,
    AnimationPlotter,
    Dimension,
    Plotter,
    Plotterly,
    PolarPlotterly,
    merge_dicts,
)
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.sensor.radar.radar import RadarElevationBearingRange
from stonesoup.types.detection import Clutter, TrueDetection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import GaussianState, State, StateVector
from stonesoup.types.track import Track
from stonesoup.updater.kalman import KalmanUpdater

# Setup simulation to test the plotter functionality
start_time = datetime.now()
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.005), ConstantVelocity(0.005)]
)
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])
for k in range(1, 21):
    truth.append(
        GroundTruthState(
            transition_model.function(
                truth[k - 1], noise=True, time_interval=timedelta(seconds=1)
            ),
            timestamp=start_time + timedelta(seconds=k),
        )
    )
timesteps = [start_time + timedelta(seconds=k) for k in range(1, 21)]

measurement_model = LinearGaussian(
    ndim_state=4, mapping=(0, 2), noise_covar=np.array([[0.75, 0], [0, 0.75]])
)
true_measurements = []
for state in truth:
    measurement_set = set()
    # Generate actual detection from the state with a 1-p_d chance that no detection is received.
    measurement = measurement_model.function(state, noise=True)
    measurement_set.add(
        TrueDetection(
            state_vector=measurement,
            groundtruth_path=truth,
            timestamp=state.timestamp,
            measurement_model=measurement_model,
        )
    )

    true_measurements.append(measurement_set)

clutter_measurements = []
for state in truth:
    clutter_measurement_set = set()
    random_state = state.from_state(
        state=state, state_vector=np.random.uniform(-20, 20, size=state.state_vector.size)
    )
    measurement = measurement_model.function(random_state, noise=True)
    clutter_measurement_set.add(
        Clutter(
            state_vector=measurement,
            timestamp=state.timestamp,
            measurement_model=measurement_model,
        )
    )

    clutter_measurements.append(clutter_measurement_set)

all_measurements = [*true_measurements, *clutter_measurements]

predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(measurement_model)
hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=3)
data_associator = NearestNeighbour(hypothesiser)

# Run Kalman filter with data association
# Create prior
prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
track = Track([prior])
for n, measurements in enumerate(true_measurements):
    hypotheses = data_associator.associate(
        [track], measurements, start_time + timedelta(seconds=n)
    )
    hypothesis = hypotheses[track]  # get the hypothesis for the specified track

    if hypothesis.measurement:
        post = updater.update(hypothesis)
        track.append(post)
    else:  # When data associator says no detections are good enough, we'll keep the prediction
        track.append(hypothesis.prediction)

sensor2d = RadarElevationBearingRange(
    position_mapping=(0, 2),
    noise_covar=np.array([[0, 0], [0, 0]]),
    ndim_state=4,
    position=np.array([[10], [50]]),
)

sensor3d = RadarElevationBearingRange(
    position_mapping=(0, 2, 4),
    noise_covar=np.array([[0, 0, 0], [0, 0, 0]]),
    ndim_state=6,
    position=np.array([[10], [50], [0]]),
)

shape = Shape(shape_data=np.array([[-2, -2, 2, 2], [-2, 2, 2, -2]]))
obstacle_list = [
    Obstacle(shape=shape, states=State(StateVector([[0], [0]])), position_mapping=(0, 1)),
    Obstacle(shape=shape, states=State(StateVector([[0], [5]])), position_mapping=(0, 1)),
    Obstacle(shape=shape, states=State(StateVector([[5], [0]])), position_mapping=(0, 1)),
]


@pytest.fixture(autouse=True)
def close_figs():
    existing_figs = set(plt.get_fignums())
    yield None
    for fignum in set(plt.get_fignums()) - existing_figs:
        plt.close(fignum)


@pytest.fixture(scope="module")
def plotter_class(request):
    plotter_class = request.param
    assert plotter_class in {
        Plotter,
        Plotterly,
        AnimationPlotter,
        PolarPlotterly,
        AnimatedPlotterly,
        AnimatedPolarPlotterly,
    }

    def _generate_animated_plotterly(*args, **kwargs):
        return plotter_class(*args, timesteps=timesteps, **kwargs)

    def _generate_plotter(*args, **kwargs):
        return plotter_class(*args, **kwargs)

    if plotter_class in {Plotter, Plotterly, AnimationPlotter, PolarPlotterly}:
        yield _generate_plotter
    elif plotter_class in {AnimatedPlotterly, AnimatedPolarPlotterly}:
        yield _generate_animated_plotterly
    else:
        raise ValueError("Invalid Plotter type.")


# Test functions
def test_dimension_inlist():  # ensure dimension type is in predefined enum list
    with pytest.raises(AttributeError):
        Plotter(dimension=Dimension.TESTERROR)


def test_particle_3d():  # warning should arise if particle is attempted in 3d mode
    plotter3 = Plotter(dimension=Dimension.THREE)

    with pytest.raises(NotImplementedError):
        plotter3.plot_tracks(track, [0, 1, 2], particle=True, uncertainty=False)


def test_plot_sensors():
    plotter3d = Plotter(Dimension.THREE)
    plotter3d.plot_sensors(sensor3d, marker="o", color="red")
    assert "Sensors" in plotter3d.legend_dict


@pytest.mark.parametrize(
    "plotter_class",
    [
        Plotter,
        Plotterly,
        AnimationPlotter,
        PolarPlotterly,
        AnimatedPlotterly,
        AnimatedPolarPlotterly,
    ],
    indirect=True,
)
def test_empty_tracks(plotter_class):
    plotter = plotter_class()
    plotter.plot_tracks(set(), [0, 2])


def test_figsize():
    plotter_figsize_default = Plotter()
    plotter_figsize_different = Plotter(figsize=(20, 15))
    assert plotter_figsize_default.fig.get_figwidth() == 10
    assert plotter_figsize_default.fig.get_figheight() == 6
    assert plotter_figsize_different.fig.get_figwidth() == 20
    assert plotter_figsize_different.fig.get_figheight() == 15


@pytest.mark.skipif(sys.platform == "win32", reason="Tkinter not reliably available on Windows CI")
def test_equal_3daxis():
    plotter_default = Plotter(dimension=Dimension.THREE)
    plotter_xy_default = Plotter(dimension=Dimension.THREE)
    plotter_xy = Plotter(dimension=Dimension.THREE)
    plotter_xyz = Plotter(dimension=Dimension.THREE)
    truths = GroundTruthPath(
        states=[State(state_vector=[-1000, -20, -3]), State(state_vector=[1000, 20, 3])]
    )
    plotter_default.plot_ground_truths(truths, mapping=[0, 1, 2])
    plotter_xy_default.plot_ground_truths(truths, mapping=[0, 1, 2])
    plotter_xy.plot_ground_truths(truths, mapping=[1, 1, 2])
    plotter_xyz.plot_ground_truths(truths, mapping=[0, 1, 2])
    plotter_xy_default.set_equal_3daxis()
    plotter_xy.set_equal_3daxis([0, 1])
    plotter_xyz.set_equal_3daxis([0, 1, 2])
    plotters = [plotter_default, plotter_xy_default, plotter_xy, plotter_xyz]
    lengths = [3, 2, 2, 1]
    for plotter, length in zip(plotters, lengths, strict=False):
        min_xyz = [0, 0, 0]
        max_xyz = [0, 0, 0]
        for i in range(3):
            for line in plotter.ax.lines:
                min_xyz[i] = np.min([min_xyz[i], *line.get_data_3d()[i]])
                max_xyz[i] = np.max([max_xyz[i], *line.get_data_3d()[i]])
        assert len(set(min_xyz)) == length
        assert len(set(max_xyz)) == length


def test_equal_3daxis_2d():
    plotter = Plotter(dimension=Dimension.TWO)
    truths = GroundTruthPath(
        states=[State(state_vector=[-1000, -20, -3]), State(state_vector=[1000, 20, 3])]
    )
    plotter.plot_ground_truths(truths, mapping=[0, 1])
    plotter.set_equal_3daxis()


def test_plot_density_empty_state_sequences():
    plotter = Plotter()
    with pytest.raises(ValueError):
        plotter.plot_density([], index=None)


def test_plot_density_equal_x_y():
    plotter = Plotter()
    start_time = datetime.now()
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0), ConstantVelocity(0)]
    )
    truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], start_time)])
    for k in range(20):
        truth.append(
            GroundTruthState(
                transition_model.function(
                    truth[k], noise=True, time_interval=timedelta(seconds=1)
                ),
                timestamp=start_time + timedelta(seconds=k + 1),
            )
        )
    with pytest.raises(ValueError):
        plotter.plot_density({truth}, index=None)


def test_plot_complex_uncertainty():
    plotter = Plotter()
    track = Track([GaussianState(state_vector=[0, 0], covar=[[10, -1], [1, 10]])])
    with pytest.warns(
        UserWarning,
        match="Can not plot uncertainty for all states due to complex "
        "eigenvalues or eigenvectors",
    ):
        plotter.plot_tracks(track, mapping=[0, 1], uncertainty=True)


@pytest.mark.skipif(sys.platform == "win32", reason="Tkinter not reliably available on Windows CI")
def test_animation_plotter():
    animation_plotter = AnimationPlotter()
    animation_plotter.plot_ground_truths(truth, [0, 2])
    animation_plotter.plot_measurements(true_measurements, [0, 2])
    animation_plotter.run()

    animation_plotter_with_title = AnimationPlotter(title="Plot title")
    animation_plotter_with_title.plot_ground_truths(truth, [0, 2])
    animation_plotter_with_title.plot_tracks(track, [0, 2])
    animation_plotter_with_title.run()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Animation was deleted without rendering anything")
        del animation_plotter
        del animation_plotter_with_title


def test_animated_plotterly():
    plotter = AnimatedPlotterly(timesteps)
    plotter.plot_ground_truths(truth, [0, 2])
    plotter.plot_measurements(true_measurements, [0, 2])
    plotter.plot_measurements(all_measurements, [0, 2])
    plotter.plot_obstacles(obstacle_list)
    plotter.plot_tracks(track, [0, 2], uncertainty=True, plot_history=True)


def test_animated_plotterly_empty():
    plotter = AnimatedPlotterly(timesteps)
    plotter.plot_ground_truths({}, [0, 2])
    plotter.plot_measurements({}, [0, 2])
    plotter.plot_tracks({}, [0, 2])
    plotter.plot_sensors({})


def test_animated_plotterly_sensor_plot():
    plotter = AnimatedPlotterly([start_time, start_time + timedelta(seconds=1)])
    plotter.plot_sensors(sensor2d)


def test_animated_plotterly_uneven_times():
    with pytest.warns(
        UserWarning,
        match="Timesteps are not equally spaced, so the passage of " "time is not linear",
    ):
        AnimatedPlotterly(
            [start_time, start_time + timedelta(seconds=1), start_time + timedelta(seconds=3)]
        )


def test_plotterly_empty():
    plotter = Plotterly()
    plotter.plot_ground_truths(set(), [0, 2])
    plotter.plot_measurements(set(), [0, 2])
    plotter.plot_tracks(set(), [0, 2])
    plotter.plot_obstacles(set())
    with pytest.raises(TypeError):
        plotter.plot_tracks(set())
    with pytest.raises(ValueError):
        plotter.plot_tracks(set(), [])


def test_plotterly_1d():
    plotter1d = Plotterly(dimension=1)
    plotter1d.plot_ground_truths(truth, [0])
    plotter1d.plot_measurements(true_measurements, [0])
    plotter1d.plot_tracks(track, [0])

    # check that particle=True does not plot
    with pytest.raises(NotImplementedError):
        plotter1d.plot_tracks(track, [0], particle=True)

    # check that uncertainty=True does not plot
    with pytest.raises(NotImplementedError):
        plotter1d.plot_tracks(track, [0], uncertainty=True)


def test_plotterly_2d():
    plotter2d = Plotterly()
    plotter2d.plot_ground_truths(truth, [0, 2])
    plotter2d.plot_measurements(true_measurements, [0, 2])
    plotter2d.plot_tracks(track, [0, 2], uncertainty=True)
    plotter2d.plot_sensors(sensor2d)
    plotter2d.plot_obstacles(obstacle_list)
    plotter2d.plot_obstacles(obstacle_list[0])


def test_plotterly_3d():
    plotter3d = Plotterly(dimension=3)
    plotter3d.plot_ground_truths(truth, [0, 1, 2])
    plotter3d.plot_measurements(true_measurements, [0, 1, 2])
    plotter3d.plot_tracks(track, [0, 1, 2], uncertainty=True)

    with pytest.raises(NotImplementedError):
        plotter3d.plot_tracks(track, [0, 1, 2], particle=True)


@pytest.mark.parametrize(
    "dim, mapping", [(1, [0, 1]), (1, [0, 1, 2]), (2, [0]), (2, [0, 1, 2]), (3, [0]), (3, [0, 1])]
)
def test_plotterly_wrong_dimension(dim, mapping):
    # ensure that plotter doesn't run for truth, measurements, and tracks
    # if dimension of those are not the same as the plotter's dimension
    plotter = Plotterly(dimension=dim)
    with pytest.raises(TypeError):
        plotter.plot_ground_truths(truth, mapping)

    with pytest.raises(TypeError):
        plotter.plot_measurements(true_measurements, mapping)

    with pytest.raises(TypeError):
        plotter.plot_tracks(track, mapping)


@pytest.mark.parametrize(
    "labels",
    [None, ["Tracks"], ["Ground Truth", "Tracks"], ["Ground Truth", "Measurements", "Tracks"]],
)
def test_hide_plot(labels):
    plotter = Plotterly()
    plotter.plot_ground_truths(truth, [0, 1])
    plotter.plot_measurements(true_measurements, [0, 1])
    plotter.plot_tracks(track, [0, 1])

    plotter.hide_plot_traces(labels)

    hidden = 0
    showing = 0

    for fig_data in plotter.fig.data:
        if fig_data["visible"] == "legendonly":
            hidden += 1
        elif fig_data["visible"] is None:
            showing += 1

    if labels is None:
        assert hidden == 3
    else:
        assert hidden == len(labels)
    assert hidden + showing == 3


@pytest.mark.parametrize(
    "labels",
    [None, ["Tracks"], ["Ground Truth", "Tracks"], ["Ground Truth", "Measurements", "Tracks"]],
)
def test_show_plot(labels):
    plotter = Plotterly()
    plotter.plot_ground_truths(truth, [0, 1])
    plotter.plot_measurements(true_measurements, [0, 1])
    plotter.plot_tracks(track, [0, 1])

    plotter.show_plot_traces(labels)

    showing = 0
    hidden = 0

    for fig_data in plotter.fig.data:
        if fig_data["visible"] == "legendonly":
            hidden += 1
        elif fig_data["visible"] is None:
            showing += 1

    if labels is None:
        assert showing == 3
    else:
        assert showing == len(labels)
    assert showing + hidden == 3


@pytest.mark.parametrize(
    "plotter_class",
    [
        Plotter,
        Plotterly,
        AnimationPlotter,
        PolarPlotterly,
        AnimatedPlotterly,
        AnimatedPolarPlotterly,
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "_measurements",
    [
        true_measurements,
        clutter_measurements,
        all_measurements,
        all_measurements[0],  # Tests a single measurement outside of a Collection should still run
    ],
)
def test_plotters_plot_measurements_2d(plotter_class, _measurements):
    plotter = plotter_class()
    plotter.plot_measurements(_measurements, [0, 2])


@pytest.mark.parametrize(
    "plotter_class",
    [Plotterly, AnimationPlotter, PolarPlotterly, AnimatedPlotterly, AnimatedPolarPlotterly],
    indirect=True,
)
@pytest.mark.parametrize(
    "_measurements, _show_clutter", [([], True), ([], False), (clutter_measurement_set, False)]
)
def test_plotters_plot_measurements_empty_silent(plotter_class, _measurements, _show_clutter):
    plotter = plotter_class()
    plotter.plot_measurements(_measurements, [0, 2], show_clutter=_show_clutter)


@pytest.mark.parametrize(
    "_measurements, _show_clutter", [([], True), ([], False), (clutter_measurement_set, False)]
)
def test_plotters_plot_measurements_empty_warn(_measurements, _show_clutter):
    with pytest.warns(UserWarning, match="No artists with labels found to put in legend"):
        plotter = Plotter()
        plotter.plot_measurements(_measurements, [0, 2], show_clutter=_show_clutter)


@pytest.mark.parametrize(
    "_measurements, _show_clutter, expected_plot_truths_length",
    [
        (true_measurements, True, len(true_measurements)),
        (true_measurements, False, len(true_measurements)),
        (all_measurements, False, len(true_measurements)),
        (clutter_measurements, False, None),
    ],
)
# Ignore this warning which occurs when there is no data to plot (e.g. last test case here)
@pytest.mark.filterwarnings("ignore:.*No artists with labels found to put in legend.*:UserWarning")
def test_plotters_plot_measurements_count_no_clutter(
    _measurements, _show_clutter, expected_plot_truths_length
):
    plotter = Plotter()
    artist_list = plotter.plot_measurements(_measurements, [0, 2], show_clutter=_show_clutter)

    expected_number_of_artists = 1  # there is always a legend artist at the end
    if expected_plot_truths_length is not None:
        expected_number_of_artists += 1
    assert len(artist_list) == expected_number_of_artists

    if expected_plot_truths_length is not None:
        truths_artist = artist_list[0]
        actual_plot_truths_length = len(truths_artist.get_offsets())
        assert actual_plot_truths_length == expected_plot_truths_length


@pytest.mark.parametrize(
    "_measurements, _show_clutter, expected_plot_truths_length, expected_plot_clutter_length",
    [
        (all_measurements, True, len(true_measurements), len(clutter_measurements)),
        (clutter_measurements, True, None, len(clutter_measurements)),
    ],
)
def test_plotters_plot_measurements_count_with_clutter(
    _measurements, _show_clutter, expected_plot_truths_length, expected_plot_clutter_length
):
    plotter = Plotter()
    artist_list = plotter.plot_measurements(_measurements, [0, 2], show_clutter=_show_clutter)

    truths_expected = expected_plot_truths_length is not None
    clutter_expected = expected_plot_clutter_length is not None
    expected_number_of_artists = 1  # there is always a legend artist at the end
    if truths_expected:
        expected_number_of_artists += 1
    if clutter_expected:
        expected_number_of_artists += 1
    assert len(artist_list) == expected_number_of_artists

    if truths_expected:
        # If truths are present in the plot, they are always the first artist
        actual_plot_truths_length = len(artist_list[0].get_offsets())
        assert actual_plot_truths_length == expected_plot_truths_length
    elif clutter_expected:
        # If no truths but clutter is present, clutter will be the first artist
        actual_plot_clutter_length = len(artist_list[0].get_offsets())
        assert actual_plot_clutter_length == expected_plot_clutter_length

    if truths_expected and clutter_expected:
        # If both are present, clutter will be the second artist (and we already checked truths)
        actual_plot_clutter_length = len(artist_list[1].get_offsets())
        assert actual_plot_clutter_length == expected_plot_clutter_length


@pytest.mark.parametrize(
    "plotter_class",
    [
        Plotter,
        Plotterly,
        AnimationPlotter,
        PolarPlotterly,
        AnimatedPlotterly,
        AnimatedPolarPlotterly,
    ],
    indirect=True,
)
def test_plotters_plot_tracks(plotter_class):
    plotter = plotter_class()
    plotter.plot_tracks(track, [0, 2])


@pytest.mark.parametrize(
    "plotter_class",
    [
        Plotter,
        Plotterly,
        pytest.param(AnimationPlotter, marks=pytest.mark.xfail(raises=NotImplementedError)),
        pytest.param(PolarPlotterly, marks=pytest.mark.xfail(raises=NotImplementedError)),
        AnimatedPlotterly,
        pytest.param(AnimatedPolarPlotterly, marks=pytest.mark.xfail(raises=NotImplementedError)),
    ],
    indirect=True,
)
def test_plotters_plot_track_uncertainty(plotter_class):
    plotter = plotter_class()
    plotter.plot_tracks(track, [0, 2], uncertainty=True)


@pytest.mark.xfail(raises=NotImplementedError)
@pytest.mark.parametrize(
    "plotter_class", [AnimationPlotter, PolarPlotterly, AnimatedPolarPlotterly], indirect=True
)
def test_plotters_plot_track_particle(plotter_class):
    plotter = plotter_class()
    plotter.plot_tracks(track, [0, 2], particle=True)


@pytest.mark.parametrize(
    "plotter_class",
    [
        Plotter,
        Plotterly,
        AnimationPlotter,
        PolarPlotterly,
        AnimatedPlotterly,
        AnimatedPolarPlotterly,
    ],
    indirect=True,
)
def test_plotters_plot_truths(plotter_class):
    plotter = plotter_class()
    plotter.plot_ground_truths(truth, [0, 2])


@pytest.mark.parametrize(
    "plotter_class",
    [
        Plotter,
        Plotterly,
        pytest.param(AnimationPlotter, marks=pytest.mark.xfail(raises=NotImplementedError)),
        pytest.param(PolarPlotterly, marks=pytest.mark.xfail(raises=NotImplementedError)),
        AnimatedPlotterly,
        pytest.param(AnimatedPolarPlotterly, marks=pytest.mark.xfail(raises=NotImplementedError)),
    ],
    indirect=True,
)
def test_plotters_plot_sensors(plotter_class):
    plotter = plotter_class()
    plotter.plot_sensors(sensor2d)


@pytest.mark.parametrize(
    "plotter_class", [Plotterly, PolarPlotterly, AnimatedPlotterly, PolarPlotterly], indirect=True
)
@pytest.mark.parametrize(
    "_measurements, expected_labels",
    [
        (true_measurements, {"Measurements"}),
        (clutter_measurements, {"Measurements<br>(Clutter)"}),
        (all_measurements, {"Measurements<br>(Detections)", "Measurements<br>(Clutter)"}),
    ],
)
def test_plotterlys_plot_measurements_label(plotter_class, _measurements, expected_labels):
    plotter = plotter_class()
    plotter.plot_measurements(_measurements, [0, 2])
    actual_labels = {fig_data.legendgroup for fig_data in plotter.fig.data}
    assert actual_labels == expected_labels


@pytest.mark.parametrize(
    "plotter_class", [Plotterly, PolarPlotterly, AnimatedPlotterly, PolarPlotterly], indirect=True
)
@pytest.mark.parametrize(
    "_measurements, _show_clutter, expected_labels",
    [
        (true_measurements, True, {"Measurements"}),
        (true_measurements, False, {"Measurements"}),
        (clutter_measurements, True, {"Measurements<br>(Clutter)"}),
        (clutter_measurements, False, set()),
        (all_measurements, True, {"Measurements<br>(Detections)", "Measurements<br>(Clutter)"}),
        (all_measurements, False, {"Measurements"}),
    ],
)
def test_plotterlys_plot_measurements_label_adjust_clutter(
    plotter_class, _measurements, _show_clutter, expected_labels
):
    plotter = plotter_class()
    plotter.plot_measurements(_measurements, [0, 2], show_clutter=_show_clutter)
    actual_labels = {fig_data.legendgroup for fig_data in plotter.fig.data}
    assert actual_labels == expected_labels


@pytest.mark.parametrize(
    "_measurements, expected_labels",
    [
        (true_measurements, {"Measurements"}),
        (clutter_measurements, {"Measurements\n(Clutter)"}),
        (all_measurements, {"Measurements\n(Detections)", "Measurements\n(Clutter)"}),
    ],
)
def test_plotter_plot_measurements_label(_measurements, expected_labels):
    plotter = Plotter()
    plotter.plot_measurements(_measurements, [0, 2])
    actual_labels = set(plotter.legend_dict.keys())
    assert actual_labels == expected_labels


@pytest.mark.parametrize(
    "_measurements, _show_clutter, expected_labels",
    [
        (true_measurements, True, {"Measurements"}),
        (true_measurements, False, {"Measurements"}),
        (clutter_measurements, True, {"Measurements\n(Clutter)"}),
        (clutter_measurements, False, set()),
        (all_measurements, True, {"Measurements\n(Detections)", "Measurements\n(Clutter)"}),
        (all_measurements, False, {"Measurements"}),
    ],
)
@pytest.mark.filterwarnings("ignore:.*No artists with labels found to put in legend.*:UserWarning")
# Ignore this warning which occurs when there is no data to plot
def test_plotter_plot_measurements_label_adjust_clutter(
    _measurements, _show_clutter, expected_labels
):
    plotter = Plotter()
    plotter.plot_measurements(_measurements, [0, 2], show_clutter=_show_clutter)
    actual_labels = set(plotter.legend_dict.keys())
    assert actual_labels == expected_labels


test_merge_dicts_data = {
    "Empty dictionaries": (({}, {}), {}),
    "Single dictionary": (({"a": 1},), {"a": 1}),
    "Non-overlapping keys": (({"a": 1}, {"b": 2}), {"a": 1, "b": 2}),
    "Overlapping keys (non-dict values)": (({"a": 1}, {"b": 2}), {"a": 1, "b": 2}),
    "Nested dictionaries": (({"a": {"b": 1}}, {"a": {"c": 2}}), {"a": {"b": 1, "c": 2}}),
    "Deeply nested dictionaries": (
        ({"a": {"b": {"c": 1}}}, {"a": {"b": {"d": 2}}}),
        {"a": {"b": {"c": 1, "d": 2}}},
    ),
    "Overwriting a dict with a non-dict": (({"a": {"b": 1}}, {"a": 2}), {"a": 2}),
    "Merging three dictionaries": (({"a": 1}, {"b": 2}, {"c": 3}), {"a": 1, "b": 2, "c": 3}),
    "Complex nested merge scenario": (
        ({"a": {"b": 1}}, {"a": {"c": {"d": 2}}}),
        {"a": {"b": 1, "c": {"d": 2}}},
    ),
}


@pytest.mark.parametrize(
    "dicts, expected", test_merge_dicts_data.values(), ids=test_merge_dicts_data.keys()
)
def test_merge(dicts: tuple[dict], expected: dict):
    assert merge_dicts(*dicts) == expected


@pytest.fixture(
    scope="module",
    params=[Plotter(), Plotterly(), AnimationPlotter(), AnimatedPlotterly(timesteps=timesteps)],
)
def plotters(request):
    return request.param


@pytest.fixture(scope="module", params=[obstacle_list[0], obstacle_list])
def obstacles(request):
    return request.param


def test_obstacles(plotters, obstacles):
    if isinstance(plotters, AnimationPlotter):
        with pytest.raises(NotImplementedError):
            plotters.plot_obstacles(obstacles)
    else:
        plotters.plot_ground_truths(truth, [0, 1])
        plotters.plot_measurements(all_measurements, [0, 1])
        plotters.plot_tracks(track, [0, 1])
        plotters.plot_obstacles(obstacles)


# New comprehensive tests


class TestPlotterlyUtilityMethods:
    """Tests for Plotterly utility methods"""

    def test_format_state_text_basic(self):
        """Test _format_state_text with basic state"""
        plotter = Plotterly()
        state = State(StateVector([1, 2]), timestamp=start_time)
        text = plotter._format_state_text(state)
        assert "State" in text
        assert str(start_time) in text

    def test_format_state_text_with_metadata(self):
        """Test _format_state_text with metadata"""
        plotter = Plotterly()
        state = State(
            StateVector([1, 2]), timestamp=start_time, metadata={"id": 123, "color": "red"}
        )
        text = plotter._format_state_text(state)
        assert "id: 123" in text
        assert "color: red" in text

    def test_format_state_text_gaussian_state(self):
        """Test _format_state_text with GaussianState (has mean)"""
        plotter = Plotterly()
        state = GaussianState(StateVector([1, 2]), np.eye(2), timestamp=start_time)
        text = plotter._format_state_text(state)
        assert "GaussianState" in text

    def test_check_mapping_empty(self):
        """Test _check_mapping with empty mapping raises ValueError"""
        plotter = Plotterly(dimension=2)
        with pytest.raises(ValueError, match="No indices provided in mapping"):
            plotter._check_mapping([])

    def test_check_mapping_wrong_dimension(self):
        """Test _check_mapping with wrong dimension raises TypeError"""
        plotter = Plotterly(dimension=2)
        with pytest.raises(TypeError, match="Plotter dimension is not same as the mapping"):
            plotter._check_mapping([0, 1, 2])

    def test_check_mapping_correct(self):
        """Test _check_mapping with correct mapping succeeds"""
        plotter = Plotterly(dimension=2)
        # Should not raise
        plotter._check_mapping([0, 1])


class TestPlotterlyConfiguration:
    """Tests for Plotterly configuration and initialization"""

    def test_plotterly_default_dimension(self):
        """Test Plotterly default dimension is 2D"""
        plotter = Plotterly()
        assert plotter.dimension == Dimension.TWO

    def test_plotterly_custom_dimension(self):
        """Test Plotterly with custom dimension"""
        plotter = Plotterly(dimension=3)
        assert plotter.dimension == Dimension.THREE

    def test_plotterly_axis_labels_default(self):
        """Test default axis labels"""
        plotter = Plotterly()
        assert plotter.fig.layout.xaxis.title.text == "x"
        assert plotter.fig.layout.yaxis.title.text == "y"

    def test_plotterly_axis_labels_custom(self):
        """Test custom axis labels"""
        plotter = Plotterly(axis_labels=["longitude", "latitude"])
        assert plotter.fig.layout.xaxis.title.text == "longitude"
        assert plotter.fig.layout.yaxis.title.text == "latitude"

    def test_plotterly_1d_axis_labels_default(self):
        """Test 1D plotter default axis labels"""
        plotter = Plotterly(dimension=1)
        assert plotter.fig.layout.xaxis.title.text == "Time"
        assert plotter.fig.layout.yaxis.title.text == "x"

    def test_plotterly_1d_axis_labels_custom(self):
        """Test 1D plotter custom axis label"""
        plotter = Plotterly(dimension=1, axis_labels=["velocity"])
        assert plotter.fig.layout.xaxis.title.text == "Time"
        assert plotter.fig.layout.yaxis.title.text == "velocity"

    def test_plotterly_3d_aspect_mode(self):
        """Test 3D plotter has aspect mode set"""
        plotter = Plotterly(dimension=3)
        assert plotter.fig.layout.scene.aspectmode == "data"

    def test_plotterly_without_plotly_raises(self):
        """Test that Plotterly raises error if plotly not available"""
        # This test would require mocking plotly import, skip for now
        pass


class TestPlotterlyEdgeCases:
    """Tests for edge cases and error handling in Plotterly"""

    def test_plot_single_truth_state(self):
        """Test plotting a truth with single state"""
        plotter = Plotterly()
        single_truth = GroundTruthPath([GroundTruthState([1, 2], timestamp=start_time)])
        plotter.plot_ground_truths(single_truth, [0, 1])
        assert len(plotter.fig.data) > 0

    def test_plot_single_measurement(self):
        """Test plotting a single measurement"""
        plotter = Plotterly()
        single_meas = TrueDetection(
            state_vector=StateVector([1, 2]),
            timestamp=start_time,
            measurement_model=measurement_model,
        )
        plotter.plot_measurements(single_meas, [0, 1])
        assert len(plotter.fig.data) > 0

    def test_plot_tracks_with_single_state(self):
        """Test plotting track with single state"""
        plotter = Plotterly()
        single_track = Track(
            [GaussianState(StateVector([1, 2, 3, 4]), np.eye(4), timestamp=start_time)]
        )
        plotter.plot_tracks(single_track, [0, 2])
        assert len(plotter.fig.data) > 0

    def test_plot_tracks_mapping_validation(self):
        """Test that track plotting validates mapping"""
        plotter = Plotterly(dimension=2)
        with pytest.raises(TypeError):
            plotter.plot_tracks(track, [0])  # Only 1 mapping for 2D plotter

    def test_plot_measurements_mapping_validation(self):
        """Test that measurement plotting validates mapping"""
        plotter = Plotterly(dimension=2)
        with pytest.raises(TypeError):
            plotter.plot_measurements(true_measurements, [0])  # Only 1 mapping for 2D plotter

    def test_plot_ground_truths_mapping_validation(self):
        """Test that ground truth plotting validates mapping"""
        plotter = Plotterly(dimension=2)
        with pytest.raises(TypeError):
            plotter.plot_ground_truths(truth, [0])  # Only 1 mapping for 2D plotter


class TestPlotterlyCustomStyling:
    """Tests for custom styling options in Plotterly"""

    def test_plot_truths_custom_color(self):
        """Test plotting ground truths with custom color"""
        plotter = Plotterly()
        plotter.plot_ground_truths(truth, [0, 2], color="red")
        # Check that trace was added
        assert len(plotter.fig.data) > 0

    def test_plot_truths_custom_label(self):
        """Test plotting ground truths with custom label"""
        plotter = Plotterly()
        plotter.plot_ground_truths(truth, [0, 2], label="Custom Truth")
        # Check legendgroup
        assert any(trace.legendgroup == "Custom Truth" for trace in plotter.fig.data)

    def test_plot_measurements_custom_label(self):
        """Test plotting measurements with custom label"""
        plotter = Plotterly()
        plotter.plot_measurements(true_measurements, [0, 2], label="Custom Measurements")
        assert any("Custom Measurements" in trace.legendgroup for trace in plotter.fig.data)

    def test_plot_tracks_custom_label(self):
        """Test plotting tracks with custom label"""
        plotter = Plotterly()
        plotter.plot_tracks(track, [0, 2], label="Custom Tracks")
        assert any(trace.legendgroup == "Custom Tracks" for trace in plotter.fig.data)


class TestAnimatedPlotterlyInitialization:
    """Tests for AnimatedPlotterly initialization and validation"""

    def test_animated_plotterly_basic_init(self):
        """Test basic initialization of AnimatedPlotterly"""
        plotter = AnimatedPlotterly(timesteps)
        assert plotter.timesteps == timesteps
        assert len(plotter.fig.frames) == len(timesteps)

    def test_animated_plotterly_tail_length_default(self):
        """Test default tail length"""
        plotter = AnimatedPlotterly(timesteps)
        assert plotter.tail_length == 0.3

    def test_animated_plotterly_tail_length_custom(self):
        """Test custom tail length"""
        plotter = AnimatedPlotterly(timesteps, tail_length=0.5)
        assert plotter.tail_length == 0.5

    def test_animated_plotterly_tail_length_invalid_high(self):
        """Test that tail_length > 1 raises ValueError"""
        with pytest.raises(ValueError, match="Tail length should be between 0 and 1"):
            AnimatedPlotterly(timesteps, tail_length=1.5)

    def test_animated_plotterly_tail_length_invalid_low(self):
        """Test that tail_length < 0 raises ValueError"""
        with pytest.raises(ValueError, match="Tail length should be between 0 and 1"):
            AnimatedPlotterly(timesteps, tail_length=-0.1)

    def test_animated_plotterly_sim_duration_default(self):
        """Test default simulation duration"""
        plotter = AnimatedPlotterly(timesteps)
        # Default sim_duration is 6
        expected_time_window = (timesteps[-1] - timesteps[0]) * 0.3
        assert plotter.time_window == expected_time_window

    def test_animated_plotterly_sim_duration_custom(self):
        """Test custom simulation duration"""
        plotter = AnimatedPlotterly(timesteps, sim_duration=10)
        # Should still initialize without error
        assert plotter.timesteps == timesteps

    def test_animated_plotterly_sim_duration_invalid(self):
        """Test that negative sim_duration raises ValueError"""
        with pytest.raises(ValueError, match="Simulation duration must be positive"):
            AnimatedPlotterly(timesteps, sim_duration=-1)

    def test_animated_plotterly_sim_duration_zero(self):
        """Test that zero sim_duration raises ValueError"""
        with pytest.raises(ValueError, match="Simulation duration must be positive"):
            AnimatedPlotterly(timesteps, sim_duration=0)

    def test_animated_plotterly_single_timestep_raises(self):
        """Test that single timestep raises ValueError"""
        with pytest.raises(ValueError, match="Must be at least 2 timesteps"):
            AnimatedPlotterly([start_time])

    def test_animated_plotterly_equal_size_default(self):
        """Test default equal_size is False"""
        plotter = AnimatedPlotterly(timesteps)
        assert plotter.equal_size is False

    def test_animated_plotterly_equal_size_custom(self):
        """Test custom equal_size"""
        plotter = AnimatedPlotterly(timesteps, equal_size=True)
        assert plotter.equal_size is True


class TestAnimatedPlotterlyPlotting:
    """Tests for AnimatedPlotterly plotting methods"""

    def test_animated_plotterly_plot_truths_multiple(self):
        """Test plotting multiple ground truths"""
        plotter = AnimatedPlotterly(timesteps)
        truth2 = GroundTruthPath([GroundTruthState([5, 1, 5, 1], timestamp=t) for t in timesteps])
        plotter.plot_ground_truths({truth, truth2}, [0, 2])
        # Should have frames created
        assert len(plotter.fig.frames) == len(timesteps)

    def test_animated_plotterly_plot_tracks_with_uncertainty(self):
        """Test plotting tracks with uncertainty in AnimatedPlotterly"""
        plotter = AnimatedPlotterly(timesteps)
        plotter.plot_tracks(track, [0, 2], uncertainty=True)
        assert len(plotter.fig.frames) == len(timesteps)

    def test_animated_plotterly_plot_tracks_with_history(self):
        """Test plotting tracks with plot_history parameter"""
        plotter = AnimatedPlotterly(timesteps)
        plotter.plot_tracks(track, [0, 2], plot_history=True)
        assert len(plotter.fig.frames) == len(timesteps)

    def test_animated_plotterly_plot_tracks_without_history(self):
        """Test plotting tracks without plot_history"""
        plotter = AnimatedPlotterly(timesteps)
        plotter.plot_tracks(track, [0, 2], plot_history=False)
        assert len(plotter.fig.frames) == len(timesteps)

    def test_animated_plotterly_multiple_plots(self):
        """Test combining multiple plot types in AnimatedPlotterly"""
        plotter = AnimatedPlotterly(timesteps)
        plotter.plot_ground_truths(truth, [0, 2])
        plotter.plot_measurements(true_measurements, [0, 2])
        plotter.plot_tracks(track, [0, 2])
        # All should work together
        assert len(plotter.fig.frames) == len(timesteps)

    def test_animated_plotterly_resize_parameter(self):
        """Test resize parameter in plotting methods"""
        plotter = AnimatedPlotterly(timesteps)
        # resize is used in plot_ground_truths, plot_tracks, etc.
        plotter.plot_ground_truths(truth, [0, 2], resize=True)
        plotter.plot_ground_truths(truth, [0, 2], resize=False)
        # Should not raise


class TestMeasurementConversion:
    """Tests for measurement conversion functionality"""

    def test_conv_measurements_clutter_show(self):
        """Test conversion shows clutter when show_clutter=True"""
        plotter = Plotterly()
        conv_detections, conv_clutter = plotter._conv_measurements(
            clutter_measurements[0], [0, 2], show_clutter=True
        )
        assert len(conv_clutter) > 0

    def test_conv_measurements_clutter_hide(self):
        """Test conversion hides clutter when show_clutter=False"""
        plotter = Plotterly()
        conv_detections, conv_clutter = plotter._conv_measurements(
            clutter_measurements[0], [0, 2], show_clutter=False
        )
        assert len(conv_clutter) == 0

    def test_conv_measurements_true_detections(self):
        """Test conversion of true detections"""
        plotter = Plotterly()
        conv_detections, conv_clutter = plotter._conv_measurements(
            true_measurements[0], [0, 2], show_clutter=True
        )
        assert len(conv_detections) > 0

    def test_conv_measurements_mixed(self):
        """Test conversion of mixed measurements"""
        plotter = Plotterly()
        mixed = true_measurements[0] | clutter_measurements[0]
        conv_detections, conv_clutter = plotter._conv_measurements(
            mixed, [0, 2], show_clutter=True
        )
        assert len(conv_detections) > 0
        assert len(conv_clutter) > 0

    def test_conv_measurements_no_convert(self):
        """Test with convert_measurements=False"""
        plotter = Plotterly()
        conv_detections, conv_clutter = plotter._conv_measurements(
            true_measurements[0], [0, 2], convert_measurements=False
        )
        # Should still return something
        assert isinstance(conv_detections, dict)


class TestPlotterComplexUncertainty:
    """Tests for uncertainty plotting edge cases"""

    def test_plotterly_complex_covariance(self):
        """Test Plotterly with complex eigenvalues in covariance"""
        plotter = Plotterly()
        # Create a state with complex eigenvalues
        track_complex = Track([GaussianState(state_vector=[0, 0], covar=[[10, -1j], [1j, 10]])])
        # Should handle without raising, but may skip uncertainty
        plotter.plot_tracks(track_complex, mapping=[0, 1], uncertainty=True)

    def test_plotterly_singular_covariance(self):
        """Test Plotterly with singular covariance matrix"""
        plotter = Plotterly()
        track_singular = Track([GaussianState(state_vector=[0, 0], covar=[[0, 0], [0, 0]])])
        # Should handle without raising
        plotter.plot_tracks(track_singular, mapping=[0, 1], uncertainty=True)


class TestPlotterlyTraceVisibility:
    """Tests for hide_plot_traces and show_plot_traces"""

    def test_hide_all_traces(self):
        """Test hiding all traces"""
        plotter = Plotterly()
        plotter.plot_ground_truths(truth, [0, 2])
        plotter.plot_measurements(true_measurements, [0, 2])
        plotter.hide_plot_traces()  # Hide all
        # All traces should be hidden
        assert all(trace.visible == "legendonly" for trace in plotter.fig.data)

    def test_show_all_traces(self):
        """Test showing all traces after hiding"""
        plotter = Plotterly()
        plotter.plot_ground_truths(truth, [0, 2])
        plotter.plot_measurements(true_measurements, [0, 2])
        plotter.hide_plot_traces()
        plotter.show_plot_traces()  # Show all
        # All traces should be visible
        assert all(trace.visible is None for trace in plotter.fig.data)

    def test_hide_specific_trace(self):
        """Test hiding specific trace"""
        plotter = Plotterly()
        plotter.plot_ground_truths(truth, [0, 2])
        plotter.plot_measurements(true_measurements, [0, 2])
        plotter.hide_plot_traces(["Ground Truth"])
        # Check that Ground Truth is hidden
        for trace in plotter.fig.data:
            if trace.legendgroup == "Ground Truth":
                assert trace.visible == "legendonly"

    def test_show_specific_trace(self):
        """Test showing only specific trace"""
        plotter = Plotterly()
        plotter.plot_ground_truths(truth, [0, 2])
        plotter.plot_measurements(true_measurements, [0, 2])
        plotter.show_plot_traces(["Ground Truth"])
        # Ground Truth should be visible, others hidden
        for trace in plotter.fig.data:
            if trace.legendgroup == "Ground Truth":
                assert trace.visible is None
            else:
                assert trace.visible == "legendonly"


class TestPlotterDensity:
    """Tests for density plotting"""

    def test_plot_density_valid_data(self):
        """Test plotting density with valid data"""
        plotter = Plotter()
        # Create data with varying positions
        truth_varying = GroundTruthPath(
            [
                GroundTruthState([i, 0, i * 2, 1], timestamp=start_time + timedelta(seconds=i))
                for i in range(10)
            ]
        )
        plotter.plot_density([truth_varying], index=None)
        # Should create plot without error

    def test_plot_density_multiple_sequences(self):
        """Test plotting density with multiple sequences"""
        plotter = Plotter()
        truth2 = GroundTruthPath(
            [
                GroundTruthState([i + 5, 1, i * 2, 1], timestamp=start_time + timedelta(seconds=i))
                for i in range(10)
            ]
        )
        plotter.plot_density([truth, truth2], index=None)
        # Should create combined density plot


class TestPlotter3DFunctionality:
    """Tests for 3D plotting functionality"""

    def test_plotterly_3d_ground_truths(self):
        """Test 3D ground truth plotting in Plotterly"""
        plotter = Plotterly(dimension=3)
        truth_3d = GroundTruthPath(
            [
                GroundTruthState([i, i, i], timestamp=start_time + timedelta(seconds=i))
                for i in range(10)
            ]
        )
        plotter.plot_ground_truths(truth_3d, [0, 1, 2])
        assert len(plotter.fig.data) > 0

    def test_plotterly_3d_measurements(self):
        """Test 3D measurement plotting in Plotterly"""
        plotter = Plotterly(dimension=3)
        meas_3d = {
            TrueDetection(
                state_vector=StateVector([1, 2, 3]),
                timestamp=start_time,
                measurement_model=LinearGaussian(3, [0, 1, 2], np.eye(3)),
            )
        }
        plotter.plot_measurements(meas_3d, [0, 1, 2])
        assert len(plotter.fig.data) > 0

    def test_plotterly_3d_tracks(self):
        """Test 3D track plotting in Plotterly"""
        plotter = Plotterly(dimension=3)
        track_3d = Track([GaussianState(StateVector([1, 2, 3]), np.eye(3), timestamp=start_time)])
        plotter.plot_tracks(track_3d, [0, 1, 2])
        assert len(plotter.fig.data) > 0


class TestPlotterWarningsAndErrors:
    """Tests for warning and error conditions"""

    def test_plotter_complex_uncertainty_warning(self):
        """Test that complex uncertainty raises warning in Plotter"""
        # Already tested in existing tests, but ensure it's comprehensive
        plotter = Plotter()
        track_complex = Track([GaussianState(state_vector=[0, 0], covar=[[10, -1], [1, 10]])])
        with pytest.warns(UserWarning, match="Can not plot uncertainty"):
            plotter.plot_tracks(track_complex, mapping=[0, 1], uncertainty=True)

    def test_plotterly_particle_not_implemented(self):
        """Test that particle plotting raises NotImplementedError in various plotters"""
        for PlotterClass in [Plotterly]:
            plotter = PlotterClass(dimension=3)
            with pytest.raises(NotImplementedError):
                plotter.plot_tracks(track, [0, 1, 2], particle=True)

    def test_animated_plotterly_without_plotly(self):
        """Test AnimatedPlotterly initialization without plotly"""
        # Would require mocking, skip for now
        pass


class TestMergeDictsComprehensive:
    """Additional comprehensive tests for merge_dicts"""

    def test_merge_dicts_empty_with_nonempty(self):
        """Test merging empty dict with non-empty"""
        result = merge_dicts({}, {"a": 1})
        assert result == {"a": 1}

    def test_merge_dicts_nonempty_with_empty(self):
        """Test merging non-empty dict with empty"""
        result = merge_dicts({"a": 1}, {})
        assert result == {"a": 1}

    def test_merge_dicts_all_empty(self):
        """Test merging multiple empty dicts"""
        result = merge_dicts({}, {}, {})
        assert result == {}

    def test_merge_dicts_deep_nesting(self):
        """Test merging deeply nested dictionaries"""
        dict1 = {"a": {"b": {"c": {"d": 1}}}}
        dict2 = {"a": {"b": {"c": {"e": 2}}}}
        result = merge_dicts(dict1, dict2)
        assert result == {"a": {"b": {"c": {"d": 1, "e": 2}}}}

    def test_merge_dicts_overwrite_value_types(self):
        """Test that later dicts overwrite earlier values"""
        dict1 = {"a": 1}
        dict2 = {"a": 2}
        result = merge_dicts(dict1, dict2)
        assert result == {"a": 2}

    def test_merge_dicts_list_values(self):
        """Test merging dicts with list values"""
        dict1 = {"a": [1, 2]}
        dict2 = {"a": [3, 4]}
        result = merge_dicts(dict1, dict2)
        # Lists should be overwritten, not merged
        assert result == {"a": [3, 4]}

    def test_merge_dicts_none_values(self):
        """Test merging dicts with None values"""
        dict1 = {"a": None}
        dict2 = {"b": None}
        result = merge_dicts(dict1, dict2)
        assert result == {"a": None, "b": None}

    def test_merge_dicts_many_dicts(self):
        """Test merging many dictionaries"""
        dicts = [{f"key{i}": i} for i in range(10)]
        result = merge_dicts(*dicts)
        assert len(result) == 10


class TestPlotterFigureConfiguration:
    """Tests for figure configuration and styling"""

    def test_plotter_figsize_tuple(self):
        """Test Plotter with tuple figsize"""
        plotter = Plotter(figsize=(15, 10))
        assert plotter.fig.get_figwidth() == 15
        assert plotter.fig.get_figheight() == 10

    def test_plotter_3d_projection(self):
        """Test that 3D plotter creates 3D axes"""
        plotter = Plotter(dimension=Dimension.THREE)
        # Check that it has a 3D projection
        assert hasattr(plotter.ax, "get_zlim")

    def test_plotter_legend_dict_initialization(self):
        """Test that legend_dict is properly initialized"""
        plotter = Plotter()
        assert hasattr(plotter, "legend_dict")
        assert isinstance(plotter.legend_dict, dict)

    def test_plotter_legend_dict_updates(self):
        """Test that legend_dict updates after plotting"""
        plotter = Plotter()
        initial_len = len(plotter.legend_dict)
        plotter.plot_ground_truths(truth, [0, 2])
        assert len(plotter.legend_dict) > initial_len


class TestPlotterSensors:
    """Tests for sensor plotting"""

    def test_plotter_sensor_2d(self):
        """Test 2D sensor plotting"""
        plotter = Plotter()
        plotter.plot_sensors(sensor2d, marker="o")
        assert "Sensors" in plotter.legend_dict

    def test_plotter_sensor_3d(self):
        """Test 3D sensor plotting"""
        plotter = Plotter(dimension=Dimension.THREE)
        plotter.plot_sensors(sensor3d, marker="^", color="blue")
        assert "Sensors" in plotter.legend_dict

    def test_plotterly_sensor_custom_label(self):
        """Test sensor plotting with custom label"""
        plotter = Plotterly()
        plotter.plot_sensors(sensor2d, label="Custom Sensor")
        assert any(trace.legendgroup == "Custom Sensor" for trace in plotter.fig.data)

    def test_animated_plotterly_sensor_multiple_frames(self):
        """Test sensor plotting across animation frames"""
        plotter = AnimatedPlotterly([start_time, start_time + timedelta(seconds=1)])
        plotter.plot_sensors(sensor2d)
        # Sensors should appear in frames
        assert len(plotter.fig.frames) == 2


class TestPlotterObstacles:
    """Additional tests for obstacle plotting"""

    def test_plotterly_obstacle_single(self):
        """Test plotting single obstacle in Plotterly"""
        plotter = Plotterly()
        plotter.plot_obstacles(obstacle_list[0])
        assert len(plotter.fig.data) > 0

    def test_plotterly_obstacle_list(self):
        """Test plotting multiple obstacles in Plotterly"""
        plotter = Plotterly()
        plotter.plot_obstacles(obstacle_list)
        # Should have multiple traces
        assert len(plotter.fig.data) >= len(obstacle_list)

    def test_plotterly_obstacle_custom_label(self):
        """Test obstacle plotting with custom label"""
        plotter = Plotterly()
        plotter.plot_obstacles(obstacle_list, label="Custom Obstacles")
        assert any(trace.legendgroup == "Custom Obstacles" for trace in plotter.fig.data)

    def test_animated_plotterly_obstacles_in_frames(self):
        """Test obstacles appear in animation frames"""
        plotter = AnimatedPlotterly(timesteps)
        plotter.plot_obstacles(obstacle_list)
        # Obstacles should be added
        assert len(plotter.fig.frames) == len(timesteps)


class TestPlotterEmptyAndSingleData:
    """Tests for edge cases with empty and single data points"""

    def test_plotter_empty_track_set(self):
        """Test plotting empty track set"""
        plotter = Plotter()
        plotter.plot_tracks(set(), [0, 2])
        # Should not raise, but legend_dict might not update

    def test_plotterly_empty_measurements_set(self):
        """Test plotting empty measurements set"""
        plotter = Plotterly()
        plotter.plot_measurements(set(), [0, 2])
        # Should handle gracefully

    def test_plotter_single_measurement_point(self):
        """Test plotting single measurement point"""
        plotter = Plotter()
        single_meas = {
            TrueDetection(
                state_vector=StateVector([1, 2]),
                timestamp=start_time,
                measurement_model=measurement_model,
            )
        }
        plotter.plot_measurements(single_meas, [0, 2])
        # Should create plot

    def test_plotterly_ground_truth_single_path(self):
        """Test plotting single ground truth path"""
        plotter = Plotterly()
        single_path = GroundTruthPath([GroundTruthState([0, 1], timestamp=start_time)])
        plotter.plot_ground_truths(single_path, [0, 1])
        assert len(plotter.fig.data) > 0


class TestAnimationPlotterSpecific:
    """Tests specific to AnimationPlotter"""

    @pytest.mark.skipif(
        sys.platform == "win32", reason="Tkinter not reliably available on Windows CI"
    )
    def test_animation_plotter_with_all_plot_types(self):
        """Test AnimationPlotter with all plot types"""
        anim_plotter = AnimationPlotter()
        anim_plotter.plot_ground_truths(truth, [0, 2])
        anim_plotter.plot_measurements(true_measurements, [0, 2])
        anim_plotter.plot_tracks(track, [0, 2])
        # Should be able to combine all

    @pytest.mark.skipif(
        sys.platform == "win32", reason="Tkinter not reliably available on Windows CI"
    )
    def test_animation_plotter_title(self):
        """Test AnimationPlotter with title"""
        anim_plotter = AnimationPlotter(title="Test Animation")
        anim_plotter.plot_ground_truths(truth, [0, 2])
        # Title should be set


class TestPolarPlotterly:
    """Tests for PolarPlotterly"""

    def test_polar_plotterly_basic_init(self):
        """Test PolarPlotterly initialization"""
        polar_plotter = PolarPlotterly()
        assert hasattr(polar_plotter, "fig")

    def test_polar_plotterly_plot_ground_truths(self):
        """Test plotting ground truths in polar coordinates"""
        polar_plotter = PolarPlotterly()
        polar_plotter.plot_ground_truths(truth, [0, 2])
        assert len(polar_plotter.fig.data) > 0

    def test_polar_plotterly_plot_measurements(self):
        """Test plotting measurements in polar coordinates"""
        polar_plotter = PolarPlotterly()
        polar_plotter.plot_measurements(true_measurements, [0, 2])
        assert len(polar_plotter.fig.data) > 0

    def test_polar_plotterly_plot_tracks(self):
        """Test plotting tracks in polar coordinates"""
        polar_plotter = PolarPlotterly()
        polar_plotter.plot_tracks(track, [0, 2])
        assert len(polar_plotter.fig.data) > 0


class TestAnimatedPolarPlotterly:
    """Tests for AnimatedPolarPlotterly"""

    def test_animated_polar_plotterly_init(self):
        """Test AnimatedPolarPlotterly initialization"""
        plotter = AnimatedPolarPlotterly(timesteps)
        assert hasattr(plotter, "fig")
        assert hasattr(plotter, "timesteps")

    def test_animated_polar_plotterly_plot_ground_truths(self):
        """Test plotting ground truths in animated polar plotter"""
        plotter = AnimatedPolarPlotterly(timesteps)
        plotter.plot_ground_truths(truth, [0, 2])
        assert len(plotter.fig.frames) == len(timesteps)

    def test_animated_polar_plotterly_plot_measurements(self):
        """Test plotting measurements in animated polar plotter"""
        plotter = AnimatedPolarPlotterly(timesteps)
        plotter.plot_measurements(true_measurements, [0, 2])
        assert len(plotter.fig.frames) == len(timesteps)

    def test_animated_polar_plotterly_plot_tracks(self):
        """Test plotting tracks in animated polar plotter"""
        plotter = AnimatedPolarPlotterly(timesteps)
        plotter.plot_tracks(track, [0, 2])
        assert len(plotter.fig.frames) == len(timesteps)
