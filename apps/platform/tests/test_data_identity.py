#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""数据身份对账测试: 确认而非发明; 三方独立取证。"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from od_platform.common import stamp
from od_platform.data_validation.identity import DataIdentity
from od_platform.data_validation.identity import reconcile_data_identity


def _di(mine, d3, d4_fp, d4_sev="PASS"):
    return DataIdentity(ok=True, data_fingerprint=mine,
                        transform_stamp={"data_fingerprint": d3} if d3 else None,
                        validate_stamp=({"data_fingerprint": d4_fp, "overall_severity": d4_sev}
                                        if d4_fp else None))

def test_reconciled_when_three_agree():
    assert _di("X", "X", "X").reconciled is True

def test_not_reconciled_when_disk_differs():
    # D6 自己算的和章里的不一样 = 数据被动过
    assert _di("Y", "X", "X").reconciled is False

def test_not_reconciled_when_a_stamp_missing():
    assert _di("X", None, "X").reconciled is False   # D3 章缺
    assert _di("X", "X", None).reconciled is False   # D4 章缺

def test_validated_pass_needs_pass():
    assert _di("X", "X", "X", "PASS").validated_pass is True
    assert _di("X", "X", "X", "ERROR").validated_pass is False


def _transform_stamp(data_fp="DATA", contract_fp="CONTRACT"):
    return {"stage": "transform", "data_fingerprint": data_fp,
            "contract_fingerprint": contract_fp, "run_id": "d3",
            "tool_version": "odp-transform/0.5"}


def _validate_stamp(severity, data_fp="DATA", contract_fp="CONTRACT"):
    return {"stage": "validate", "data_fingerprint": data_fp,
            "contract_fingerprint": contract_fp, "overall_severity": severity,
            "run_id": "d4", "tool_version": "odp-validate/0.5"}


def _reconcile(d4, *, d3=None, require_validated=True):
    d3 = _transform_stamp() if d3 is None else d3
    with patch("od_platform.data_validation.identity._recompute_data_fingerprint",
               return_value=("DATA", "CONTRACT", None)), \
         patch.object(stamp, "read_stamp", side_effect=[d3, d4]):
        return reconcile_data_identity(Path("dataset.yaml"),
                                       require_validated=require_validated)


@pytest.mark.parametrize("severity", ["PASS", "INFO", "WARNING"])
def test_trainable_severities_pass_gate_and_are_audited(severity):
    identity = _reconcile(_validate_stamp(severity))
    assert identity.ok is True
    assert identity.validation_severity == severity
    assert identity.to_dict()["validation_severity"] == severity


def test_error_severity_blocks_training():
    assert _reconcile(_validate_stamp("ERROR")).ok is False


def test_missing_validate_stamp_blocks_training():
    assert _reconcile(None).ok is False


def test_data_and_contract_mismatch_block_training():
    assert _reconcile(_validate_stamp("PASS", data_fp="OTHER")).ok is False
    assert _reconcile(_validate_stamp("PASS", contract_fp="OTHER")).ok is False


@pytest.mark.parametrize("severity", [None, "UNKNOWN", "", 1])
def test_invalid_or_unknown_severity_blocks_training(severity):
    assert _reconcile(_validate_stamp(severity)).ok is False


def test_skip_validate_semantics_are_preserved():
    assert _reconcile(None, require_validated=False).ok is True
    assert _reconcile(_validate_stamp("ERROR"), require_validated=False).ok is True
    assert _reconcile(_validate_stamp("UNKNOWN"), require_validated=False).ok is False

def test_audit_records_confirmation_not_copy():
    """审计记'确认章存在', 不整份转抄章内容(阶段 4.4)。"""
    d = _di("X", "X", "X").to_dict()
    assert d["transform_stamp_seen"] is True and d["validate_stamp_seen"] is True
    assert d["data_fingerprint"] == "X" and d["reconciled"] is True
    assert d["validation_severity"] == "PASS"
    # 不应把整枚章 dict 塞进审计
    assert "transform_stamp" not in d and "validate_stamp" not in d
