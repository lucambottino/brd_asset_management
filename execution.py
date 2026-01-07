from base import Broker


class Execution:
    def __init__(self):
        self.broker = Broker()

    def get_remaining_cash(self, available_cash):
        df_deals = self.broker.get_deals()
        df_deals["finantial"] = df_deals["volume"] * df_deals["price"]
        total_finantial = df_deals["finantial"].sum()
        remaining_cash = available_cash - total_finantial
        return remaining_cash


    def allocate_lft(self, available_cash):
        quantity_lft = 0
        remaining_cash = self.get_remaining_cash(available_cash)
        info = self.broker.get_info("LFTS11")

        if remaining_cash >= 0 and info is not None:
            price_lft = info["ask"]
            quantity_lft = int(remaining_cash // price_lft)
            remaining_cash -= quantity_lft * price_lft
            if quantity_lft > 0:
                self.broker.send_market_order("LFTS11", "BUY", quantity_lft)
                print(f"Allocated {quantity_lft} LFTS11 at price {price_lft:.2f}, remaining cash: {remaining_cash:.2f}")
            else:
                print("Not enough cash to allocate any LFTS11.")
        elif remaining_cash < 0 and info is not None:
            price_lft = info["bid"]
            quantity_lft = int(-remaining_cash // price_lft)
            remaining_cash += quantity_lft * price_lft
            if quantity_lft > 0:
                self.broker.send_market_order("LFTS11", "SELL", quantity_lft)
                print(f"Deallocated {quantity_lft} LFTS11 at price {price_lft:.2f}, remaining cash: {remaining_cash:.2f}")
            else:
                print("No LFTS11 to deallocate.")
        else:
            print("No info available for LFTS11, cannot allocate/deallocate.")

    
    def get_total_AUM(self):
        df_pos = self.broker.get_acc_position()
        print(df_pos)
        df_deals = self.broker.get_deals()

        total_AUM = 0.0
        for ticker in df_pos['ticker'].unique():
            pos_volume = df_pos.loc[df_pos['ticker'] == ticker, 'volume'].sum()
            if pos_volume == 0:
                continue

            info = self.broker.get_info(ticker)
            if info is None:
                continue

            current_price = info['bid'] if pos_volume > 0 else info['ask']
            position_value = pos_volume * current_price
            total_AUM += position_value

        return total_AUM





if __name__ == "__main__":
    available_cash = 853.94
    execution = Execution()
    execution.get_total_AUM()