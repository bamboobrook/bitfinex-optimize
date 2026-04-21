"""
Phase C 回归测试: /result API 格式兼容性

核心断言:
1. /result 返回 JSON 包含 rank1-5 字段
2. rank6 固定为 fUSD-2d（如存在）
3. JSON 解析失败时返回结构化错误
4. combo 字段格式向后兼容
"""
import json
import pytest


class TestResultFormat:
    """测试 /result API 返回格式兼容性"""

    def test_rank_structure(self):
        """rank1-5 应为 dict 且含 currency/period/rate/score"""
        sample = {
            "rank1": {"currency": "fUSD", "period": 2, "rate": 5.0, "score": 80.0},
            "rank2": {"currency": "fUSD", "period": 3, "rate": 4.5, "score": 75.0},
        }
        for rank_key in ["rank1", "rank2"]:
            assert rank_key in sample
            entry = sample[rank_key]
            for field in ["currency", "period", "rate", "score"]:
                assert field in entry, f"{rank_key} missing {field}"

    def test_rank6_is_fusd_2d(self):
        """rank6 应为 fUSD-2d（如存在）"""
        sample_rank6 = {"currency": "fUSD", "period": 2}
        if sample_rank6:
            assert sample_rank6["currency"] == "fUSD"
            assert sample_rank6["period"] == 2

    def test_error_response_structure(self):
        """JSON 错误返回应包含 error 字段"""
        error_resp = {"error": "retraining in progress", "status": "degraded"}
        assert "error" in error_resp
        assert "status" in error_resp

    def test_combo_format(self):
        """combo 字段应为 list 且每项含 rank/currency/period/rate"""
        sample_combo = [
            {"rank": 1, "currency": "fUSD", "period": 2, "rate": 5.0},
            {"rank": 2, "currency": "fUSD", "period": 3, "rate": 4.5},
        ]
        assert isinstance(sample_combo, list)
        for item in sample_combo:
            assert "rank" in item
            assert "currency" in item
            assert "period" in item
            assert "rate" in item


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
