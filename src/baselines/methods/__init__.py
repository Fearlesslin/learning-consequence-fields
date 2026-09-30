from .ca_edf import run_ca_edf
from .casds import run_casds
from .context_aware_gd import run_context_aware_gd
from .hybrid_scheduling import run_hybrid_scheduling
from .mc_flex import run_mc_flex
from .slack_time_management import run_slack_time_management
from .workload_aware_mc import run_workload_aware_mc


METHOD_RUNNERS = {
    "mc_flex": run_mc_flex,
    "workload_aware_mc": run_workload_aware_mc,
    "ca_edf": run_ca_edf,
    "context_aware_gd": run_context_aware_gd,
    "slack_time_management": run_slack_time_management,
    "casds": run_casds,
    "hybrid_scheduling": run_hybrid_scheduling,
}
