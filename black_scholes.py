import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

class BlackScholesCalc:

    @staticmethod
    def black_scholes_price(spot_price, strike_price, years_to_expiration,
                            annualized_rf_rate, sigma, option_type="call"):
        # Calculates European Option Price using Black Scholes

        if years_to_expiration <= 0:
            if option_type == "call":
                return max(spot_price - strike_price, 0)
            else:
                return max(strike_price - spot_price, 0)

        """
        d1 can be thought of as measure related to the risk-adjusted expected
        growth of the underlying asset price and influences the option's delta
        (sensitivity to the underlying price)
        """
        d1 = (np.log(spot_price / strike_price) + (annualized_rf_rate + 0.5 *
            sigma**2) * years_to_expiration) / (sigma * np.sqrt(
            years_to_expiration))
        """
        d2 is the risk-adjusted probability that the option will be exercised
        (finishes in the money)
        """
        d2 = d1 - sigma * np.sqrt(years_to_expiration)

        # Returning price of option
        if option_type == "call":
            # Calculates expected benefit from owning the stock discounted by
            # the probability of finishing in the money minus the discounted
            # strike price weighted by the probability of exercising the option
            return spot_price * norm.cdf(d1) - strike_price * np.exp(
                    -annualized_rf_rate * years_to_expiration) * norm.cdf(d2)
        else:
            # For a put option: Calculates discounted strike price weighted by
            # the probability of exercising the option minus the expected loss
            # from the stock weighted by the probability of finishing in the $
            return strike_price * np.exp(-annualized_rf_rate
                    * years_to_expiration) * norm.cdf(-d2) - (spot_price
                    * norm.cdf(-d1))

    @staticmethod
    def implied_volatility(market_price, spot_price, strike_price,
                            years_to_expiration, annualized_rf_rate,
                            option_type="call"):
        if years_to_expiration <= 0:
            return 0
