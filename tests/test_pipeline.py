 

from src.pipeline import build_pipelines
from src.train import load_config

def test_build_pipelines_retourne_deux_pipelines():
    config = load_config()
    pipeline_rf, pipeline_logreg = build_pipelines(config)
    assert pipeline_rf is not None , "on espere le pipeline n'est pas vide "
    assert pipeline_logreg is not None , "on espere le pipeline n'est pas vide "

