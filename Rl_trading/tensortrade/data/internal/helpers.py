
import operator


from .wallet import create_wallet_source

from tensortrade.data import DataFeed, Reduce, Condition
from tensortrade.wallets import Portfolio

#In the default settings, this feed is needed to compute performance of a portfolio
def create_internal_feed(portfolio: 'Portfolio'):

    base_symbol = portfolio.base_instrument.symbol
    sources = []

    for wallet in portfolio.wallets:
        symbol = wallet.instrument.symbol
        sources += [create_wallet_source(wallet, include_worth=(symbol != base_symbol))]

    worth_nodes = Condition(
        "worths",
        lambda node: node.name.endswith(base_symbol + ":/total") or node.name.endswith("worth")
    )(*sources)
    
    worth_nodes.build()

    net_worth = Reduce("net_worth", func=operator.add)(worth_nodes)

    sources += [net_worth]

    feed = DataFeed(sources)
    feed.attach(portfolio)  #internal feed 统计账户资产信息，这部分信息在feed.next()时会通过出发portfolio.on_next()传递给portfolio，用来计算performance

    return feed
