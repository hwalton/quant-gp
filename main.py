from a_data.format_data import main as format_data_main
from b_log_fit.log_fit import main as log_fit_main
from c_gp_fit.online_gp_fit import main as online_gp_fit_main
from d_optimise_portfolio.optimise_portfolio import main as optimise_portfolio_main

def main():
    format_data_main()
    log_fit_main()
    online_gp_fit_main()
    optimise_portfolio_main()

if __name__ == "__main__":
    main()
