from scripts.task2_forecasting import run_pipeline


def test_pipeline_runs_arima_only():
    metrics = run_pipeline(model_choice="arima")

    assert not metrics.empty
    assert "Ticker" in metrics.columns
    assert "MAE" in metrics.columns
