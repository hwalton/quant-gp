from b_log_fit.log_fit import Config as Log_fit_config, load_data
from c_gp_fit.online_gp_fit import Config as Online_gp_fit_config
from d_optimise_portfolio.optimise_portfolio import Config as Optimise_portfolio_config

import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'a_data', 'bitcoin_combined_weekly_data.csv')


@dataclass(frozen=True)
class Config:
    start_datetime: str = '2015-01-01'
    end_datetime: str = '2023-10-01'
    
def main(cfg: Config = Config()):

    
    log_fit_config = Log_fit_config()
    online_gp_fit_config = Online_gp_fit_config()
    optimise_portfolio_config = Optimise_portfolio_config()



    # Here you can call the main functions of each module if needed
    # For example:
    # log_fit_main(log_fit_config)


if __name__ == "__main__":
    main()