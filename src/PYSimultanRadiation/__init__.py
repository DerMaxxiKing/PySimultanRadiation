from .logger import logger
from PySimultan import TemplateParser, DataModel, Template, yaml
from .geometry.extended_geometry import ExtendedVertex, ExtendedEdge, ExtendedEdgeLoop, ExtendedFace, ExtendedVolume
from .foi import ReferenceList
from .weather import Weather

TemplateParser.geo_bases['vertex'] = ExtendedVertex
TemplateParser.geo_bases['edge'] = ExtendedEdge
TemplateParser.geo_bases['edge_loop'] = ExtendedEdgeLoop
TemplateParser.geo_bases['face'] = ExtendedFace
TemplateParser.geo_bases['volume'] = ExtendedVolume

TemplateParser.bases['ReferenceList'] = ReferenceList
TemplateParser.bases['Weather'] = Weather
