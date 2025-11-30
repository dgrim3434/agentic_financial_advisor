from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any




# ============================================
# USER PROFILE
# ============================================


@dataclass
class UserProfile:
    name: Optional[str] = None
    age: Optional[int] = None


    risk_tolerance: Optional[str] = None          # low | medium | high
    investment_horizon: Optional[str] = None      # short | medium | long
    experience_level: Optional[str] = None        # beginner | intermediate | advanced


    preferred_sectors: List[str] = field(default_factory=list)
    ticker_watchlist: List[str] = field(default_factory=list)
    constraints: Optional[str] = None
    notes: Optional[str] = None
    investment_amount: Optional[float] = None
    def is_complete(self) -> bool:
        """
        Minimum completeness test.
        Agents may choose to ask follow-ups if missing fields.
        """
        required = [self.risk_tolerance, self.investment_horizon, self.investment_amount]
        return all(v is not None for v in required)





# SEC ANALYSIS OUTPUT MODEL



@dataclass
class SecAnalysisResult:
    ticker: str
    question: str
    answer: str
    sources: List[Dict[str, Any]]  # SEC metadata





# QUANT / MARKET DATA MODEL (future)



@dataclass
class QuantMetrics:
    ticker: str
    volatility_1y: Optional[float] = None
    max_drawdown: Optional[float] = None
    momentum_score: Optional[float] = None
    trend: Optional[str] = None     # uptrend | downtrend | flat





# SENTIMENT ANALYSIS MODEL



@dataclass
class SentimentSummary:
    ticker: str
    overall_sentiment: Optional[str] = None
    themes: List[str] = field(default_factory=list)
    articles: List[Dict[str, Any]] = field(default_factory=list)





# STRATEGY MODEL


@dataclass
class StrategyRecommendation:
    recommendation: str
    reasoning: str
    cited_data: Dict[str, Any] = field(default_factory=dict)
    risk_considerations: List[str] = field(default_factory=list)
    alternatives: Dict[str, str] = field(default_factory=dict)





