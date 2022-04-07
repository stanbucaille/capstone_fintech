
import operator

from tensortrade.data import Lambda, Module, Select, BinOp
from tensortrade.wallets import Wallet

#base_symbol: str, 
def create_wallet_source(wallet: Wallet, include_worth=True):
    exchange_name = wallet.exchange.name
    symbol = wallet.instrument.symbol

    #用with的方式建立Module会调用__enter__,把自己放进Module.CONTEXTS里面,之后所有生成的node在初始化的时候都会把自己纳入到CONTEXTS[-1]的inputs中
    with Module(exchange_name + ":/" + symbol) as wallet_ds:
        free_balance = Lambda("free", lambda w: w.balance.size, wallet)
        locked_balance = Lambda("locked", lambda w: w.locked_balance.size, wallet)
        total_balance = Lambda("total", lambda w: w.total_balance.size, wallet)


    if include_worth:
        #__call__这个select的时候传list不行，因为__call__的argument是*args，这样list外面会套一个tuple
        price = Select(lambda node: node.name.startswith(symbol), symbol)(wallet.exchange)  
        #尽管这里的price是static node, 但是如果包含在某个datafeed的话，在调用next()时会调用其inputs中隐含动态node的更新方法（一般是forward）
        #I don't want the price to appear in the output of internal_feed.next(), but the price is embedded in the worth nodes, hence it will also be affected by datafeed.next()
        worth = BinOp("worth", operator.mul)(price, total_balance)  #need to delete select node inside it
      
        wallet_ds.add_node(worth)
    
    return wallet_ds
