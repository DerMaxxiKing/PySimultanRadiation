from PySimultan import TemplateParser, DataModel, Template, yaml
from .geometry.extended_geometry import ExtendedVertex, ExtendedEdge, ExtendedEdgeLoop, ExtendedFace, ExtendedVolume
import colorlog
from .foi import ReferenceList
from .weather import Weather

TemplateParser.geo_bases['vertex'] = ExtendedVertex
TemplateParser.geo_bases['edge'] = ExtendedEdge
TemplateParser.geo_bases['edge_loop'] = ExtendedEdgeLoop
TemplateParser.geo_bases['face'] = ExtendedFace
TemplateParser.geo_bases['volume'] = ExtendedVolume

TemplateParser.bases['ReferenceList'] = ReferenceList
TemplateParser.bases['Weather'] = Weather

handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

handler.setFormatter(formatter)

logger = colorlog.getLogger('PySimultanRadiation')
logger.addHandler(handler)
