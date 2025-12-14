from stonesoup.buffered_generator import BufferedGenerator


class TestBuffer(BufferedGenerator):
    @BufferedGenerator.generator_method
    def create_numbers(self):
        yield from range(10)


def test_generation():
    test = TestBuffer()
    for expected, actual in zip(range(10), test, strict=False):
        assert expected == actual


def test_buffering():
    test = TestBuffer()
    for expected, actual in zip(range(10), (test.current for _ in test), strict=False):
        assert expected == actual
