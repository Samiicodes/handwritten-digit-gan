'''Tests for number formatting in the report.'''

from digitgan.evaluation import MetricSummary
from digitgan.reporting import format_fid, format_mse, format_rate, format_summary


def test_a_single_run_shows_only_the_value() -> None:
    assert format_summary(MetricSummary(0.4129, 0.0), format_mse, seeds=1) == '0.413'


def test_several_runs_show_the_spread() -> None:
    assert format_summary(MetricSummary(12.345, 1.2), format_fid, seeds=3) == '12.35 ± 1.20'


def test_percentages_carry_one_sign_after_the_spread() -> None:
    assert format_summary(MetricSummary(0.985, 0.003), format_rate, seeds=3) == '98.5 ± 0.3%'


def test_missing_metrics_are_marked_not_applicable() -> None:
    assert format_summary(None, format_rate, seeds=3) == 'n/a'
