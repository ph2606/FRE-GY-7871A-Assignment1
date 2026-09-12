"""Boundary and denominator checks for Item 1A removal."""
import pytest

from src.parse import html_to_text, tokenize
from src.risk_sections import exclude_item1a


def test_linked_contents_and_inline_reference_do_not_remove_business():
    filing = '''<div><p><a href="#risk">Item 1A. Risk Factors</a></p>
    <p><a href="#next">Item 1B. Unresolved Staff Comments</a></p>
    <p>Item 1. Business</p><p>Our business may grow. See <b>Item 1A. Risk Factors</b>.</p>
    <p id="risk">Item 1A. Risk Factors</p><p>Our customers could leave. Risk may increase.</p>
    <p id="next">Item 1B. Unresolved Staff Comments</p><p>None.</p></div>'''
    result = exclude_item1a(filing, "10-K")
    assert result["status"] == "found"
    assert "Our business may grow" in result["text"]
    assert "Our customers could leave" not in result["text"]
    assert 'Item 1B. Unresolved Staff Comments None.' in result["text"]
    assert result["original_n_words"] == len(tokenize(html_to_text(filing)))
    assert result["original_n_words"] == result["retained_n_words"] + result["removed_n_words"]


def test_split_table_heading_and_numeric_table_filter_match_baseline():
    filing = '''<p>Item 1. Business</p><p>The company may grow.</p>
    <table><tr><td><b>ITEM 1A.</b></td><td><b>RISK FACTORS</b></td></tr></table>
    <p>Demand could fall.</p><table><tr><td>123456789</td><td>May</td></tr></table>
    <p>ITEM 1C. CYBERSECURITY</p><p>Our controls may improve.</p>'''
    result = exclude_item1a(filing, "10-K")
    assert result["status"] == "found"
    assert result["end_heading"] == "ITEM 1C. CYBERSECURITY"
    assert "Demand could fall" not in result["text"]
    assert "Our controls may improve" in result["text"]
    assert result["original_n_words"] == len(tokenize(html_to_text(filing)))


def test_quarterly_short_no_change_section_is_removed():
    filing = '''<p>Item 1. Legal Proceedings</p><p>Nothing material.</p>
    <p>Item 1A. Risk Factors</p><p>No material changes.</p>
    <p>Item 2. Unregistered Sales of Equity Securities and Use of Proceeds</p><p>None.</p>'''
    result = exclude_item1a(filing, "10-Q")
    assert result["status"] == "found"
    assert "No material changes" not in result["text"]
    assert result["removed_n_words"] == 6


def test_quarterly_absence_requires_body_sequence():
    filing = '''<p>Item 1. Legal Proceedings</p><p>Nothing material.</p>
    <p>Item 2. Unregistered Sales of Equity Securities</p><p>None.</p>'''
    result = exclude_item1a(filing, "10-Q")
    assert result["status"] == "absent"
    assert result["text"] == html_to_text(filing)
    assert result["removed_n_words"] == 0
    assert exclude_item1a("<p>Could may risk.</p>", "10-Q")["status"] == "ambiguous"


def test_missing_end_is_ambiguous_and_not_zero():
    result = exclude_item1a("<p>Item 1A. Risk Factors</p><p>Could may risk.</p>", "10-K")
    assert result["status"] == "ambiguous"
    assert result["text"] is None
    assert result["removed_n_words"] is None


def test_unlinked_contents_page_numbers_are_not_body():
    filing = '''<p>Item 1A. Risk Factors</p><p>12</p><p>Item 1B. Unresolved Staff Comments</p><p>20</p>
    <p>Item 1. Business</p><p>Our uncertainty may grow.</p><p>Item 1A. Risk Factors</p>
    <p>We could lose customers.</p><p>Item 1B. Unresolved Staff Comments</p><p>None.</p>'''
    result = exclude_item1a(filing, "10-K")
    assert result["status"] == "found"
    assert "Our uncertainty may grow" in result["text"]
    assert "We could lose customers" not in result["text"]


def test_competing_substantial_sections_are_not_guessed():
    section = "<p>Item 1A. Risk Factors</p><p>" + "Risk could grow. " * 20 + "</p><p>Item 2. Properties</p>"
    assert exclude_item1a(section + section, "10-K")["status"] == "ambiguous"


def test_hidden_facts_do_not_change_excluded_denominator():
    filing = '''<p>Our business may improve.</p><ix:hidden><p>Item 1A. Risk Factors</p>Hidden Risk</ix:hidden>
    <p>Item 1A. Risk Factors</p><ix:nonNumeric>Demand could fall.</ix:nonNumeric>
    <p>Item 2. Properties</p><p>Visible uncertain developments.</p>'''
    result = exclude_item1a(filing, "10-K")
    assert result["status"] == "found"
    assert result["original_n_words"] == len(tokenize(html_to_text(filing)))


def test_numeric_end_heading_is_preserved_without_removing_next_section():
    filing = '''<p>Our business may improve.</p><p>Item 1A. Risk Factors</p><p>Demand could fall.</p>
    <table><tr><td>Item 2. Unregistered Sales</td><td>123456789123456789</td></tr></table>
    <p>We may repurchase shares.</p><p>Item 3. Defaults Upon Senior Securities</p><p>None.</p>'''
    result = exclude_item1a(filing, "10-Q")
    assert result["status"] == "found"
    assert result["end_heading"] == "Item 2. Unregistered Sales"
    assert "We may repurchase shares" in result["text"]
    assert result["original_n_words"] == len(tokenize(html_to_text(filing)))


def test_only_original_forms_are_accepted():
    with pytest.raises(ValueError):
        exclude_item1a("<p>May.</p>", "10-K/A")


@pytest.mark.parametrize('label', ['Item 1(A).', 'Item 1.A.', 'It em 1 A.'])
def test_heading_label_variants_and_split_end_title(label):
    raw = '<p>Item 1. Business</p><p>Our business may grow.</p><p>'+label+' Ris k Factors</p><p>Demand could fall.</p><p>Item 1.B. Unres olved Staff Comments</p><p>None.</p>'
    r = exclude_item1a(raw, '10-K')
    assert r['status'] == 'found'
    assert 'Demand could fall' not in r['text']
    assert 'Our business may grow' in r['text']
    assert 'Unres olved Staff Comments' in r['text']


def test_unnumbered_risk_heading_uses_document_contents_anchor():
    raw = '''<p><a href="#risk">Risk Factors</a></p><p><a href="#legal">Legal Proceedings</a></p>
    <p>Our business may grow.</p><div id="risk"></div><p>Table of Contents</p>
    <h2>RISK FACTORS</h2><p>Demand could fall.</p><div id="legal"></div>
    <h2>LEGAL PROCEEDINGS</h2><p>A case may settle.</p>'''
    r = exclude_item1a(raw,'10-K')
    assert r['status']=='found'
    assert 'Demand could fall' not in r['text']
    assert 'A case may settle' in r['text']


def test_partial_linked_contents_row_cannot_start_section():
    raw = '''<table><tr><td>Item 1A.</td><td>Risk Factors</td><td><a href="#risk">20</a></td></tr></table>
    <p>Item 1. Business</p><p>Our business may grow.</p><p id="risk">Item 1A. Risk Factors</p>
    <p>Demand could fall.</p><p>Item 1B. Unresolved Staff Comments</p><p>None.</p>'''
    r=exclude_item1a(raw,'10-K')
    assert r['status']=='found'
    assert 'Our business may grow' in r['text']
