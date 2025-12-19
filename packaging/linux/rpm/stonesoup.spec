%global srcname stonesoup
%global sum Framework for target tracking and state estimation

Name:           python-%{srcname}
Version:        1.0.0
Release:        1%{?dist}
Summary:        %{sum}

License:        MIT
URL:            https://github.com/dstl/Stone-Soup
Source0:        %{url}/archive/v%{version}/%{srcname}-%{version}.tar.gz

BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  python3-setuptools
BuildRequires:  python3-numpy
BuildRequires:  python3-scipy
BuildRequires:  python3-matplotlib
BuildRequires:  cmake

%description
Stone Soup is a software project to provide the target tracking and state
estimation community with a framework for the development and testing of
tracking and state estimation algorithms.

%package -n python3-%{srcname}
Summary:        %{sum}
Requires:       python3-numpy
Requires:       python3-scipy
Requires:       python3-matplotlib
Requires:       python3-ruamel-yaml
%{?python_provide:%python_provide python3-%{srcname}}

%description -n python3-%{srcname}
Stone Soup is a software project to provide the target tracking and state
estimation community with a framework for the development and testing of
tracking and state estimation algorithms.

This package contains the Python 3 library.

%package -n python3-%{srcname}-doc
Summary:        Documentation for %{name}
BuildArch:      noarch

%description -n python3-%{srcname}-doc
Documentation for Stone Soup framework.

%prep
%autosetup -n Stone-Soup-%{version}

%build
%py3_build

# Build documentation if sphinx is available
if command -v sphinx-build &> /dev/null; then
    pushd docs
    make html
    popd
fi

%install
%py3_install

# Install documentation
if [ -d docs/build/html ]; then
    mkdir -p %{buildroot}%{_docdir}/python3-%{srcname}
    cp -r docs/build/html %{buildroot}%{_docdir}/python3-%{srcname}/
fi

%check
# Run tests
# Uncomment when ready
# %{__python3} -m pytest stonesoup

%files -n python3-%{srcname}
%license LICENSE
%doc README.md
%{python3_sitelib}/%{srcname}/
%{python3_sitelib}/%{srcname}-%{version}-py%{python3_version}.egg-info/

%files -n python3-%{srcname}-doc
%doc README.md
%if 0%{?with_docs}
%{_docdir}/python3-%{srcname}/html/
%endif

%changelog
* Thu Dec 11 2025 Stone Soup Developers <stonesoup@dstl.gov.uk> - 1.0.0-1
- Initial RPM package release
- Framework for target tracking and state estimation
- Modular component-based architecture
- Support for Kalman, particle, and other tracking algorithms

* Mon Jan 01 2024 Stone Soup Developers <stonesoup@dstl.gov.uk> - 0.1~beta-1
- Initial beta release
- Core tracking framework implementation
- Basic Kalman filter support
- Particle filter implementation
