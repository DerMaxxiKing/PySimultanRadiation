# PySimultanRadiation

Software to calculate solar irradiation for SIMULTAN-Projects

## Installation

### Windows

1. Install Docker-Desktop:  https://www.docker.com/products/docker-desktop
2. Download and install Python 3.8: https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe
3. Install PySimultanRadiation: if Python 3.8 is your default version run: pip install PySimultanRadiation in cmd-window

### Ubuntu
1. Install Docker: see https://docs.docker.com/engine/install/
2. Install Python 3.8: <br />
`sudo apt update`<br />
`sudo apt install software-properties-common`<br />
`sudo add-apt-repository ppa:deadsnakes/ppa`<br />
`sudo apt install python3.8`<br />
3. Install PySimultanRadiation: <br />
`pip3 install PySimultanRadiation`


# Usage

A shading analysis for PySimultanRadiation is defined by crating a ShadingAnalysis component in Simultan.
A Template for this component can be downloaded here: 
