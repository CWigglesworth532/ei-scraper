import unittest

from match_suppliers_v2 import classify_name_candidates


class LegalFormHeuristicRegressionTests(unittest.TestCase):
    """SKO-004 regression catalogue for country-aware legal-form candidate signals.

    These tests govern candidate generation only. A positive heuristic is not
    proof that an organisation is a social-economy organisation.
    """

    def assert_coop(self, country, name):
        result = classify_name_candidates(country, name)
        self.assertEqual(
            result["name_coop_candidate"],
            "YES",
            msg=f"Expected cooperative candidate for {country}: {name}",
        )
        self.assertIn("coop", result["name_candidate_reason"].split(","))

    def assert_not_coop(self, country, name):
        result = classify_name_candidates(country, name)
        self.assertEqual(
            result["name_coop_candidate"],
            "NO",
            msg=f"Unexpected cooperative candidate for {country}: {name}",
        )

    # Live Telos diagnostic regressions — Spain.
    def test_es_soc_dot_coop(self):
        self.assert_coop("ES", "COLECTIVO CLARIS, SOC.COOP.")

    def test_es_s_coop(self):
        self.assert_coop("ES", "CASINTRA S COOP")

    def test_es_sdad_dot_coop(self):
        self.assert_coop("ES", "COFARES SDAD.COOP.FCA.ESPAÑOLA")

    def test_es_sdad_coop_andaluza(self):
        self.assert_coop("ES", "GRUPO BIDAFARMA SDAD COOP ANDALUZA")

    def test_es_sdad_dot_coop_with_space(self):
        self.assert_coop("ES", "BUREBA-EBRO SDAD. COOP")

    def test_es_sociedad_coop(self):
        self.assert_coop("ES", "SOCIEDAD COOP NUEVAS INICIATIVAS")

    # Cross-country positives prove this is not a Spain-only patch.
    def test_de_registered_cooperative_abbreviation(self):
        self.assert_coop("DE", "Beispiel Energie eG")

    def test_de_punctuated_registered_cooperative_abbreviation(self):
        self.assert_coop("DE", "Beispiel Energie e.G.")

    def test_it_soc_coop(self):
        self.assert_coop("IT", "Impresa Verde Soc. Coop.")

    def test_pt_crl(self):
        self.assert_coop("PT", "Cooperativa Exemplo, C.R.L.")

    def test_be_cv(self):
        self.assert_coop("BE", "Voorbeeld Coöperatie CV")

    def test_nl_cooperatie_ua(self):
        self.assert_coop("NL", "Coöperatie Voorbeeld U.A.")

    # Precision / country-alignment safeguards.
    def test_de_heg_initials_are_not_eg(self):
        self.assert_not_coop("DE", "H.E.G. Haus- und Energietechnik")

    def test_de_eg_inside_word_is_not_legal_form(self):
        self.assert_not_coop("DE", "Mega Energie GmbH")

    def test_spanish_alias_does_not_fire_without_spanish_country(self):
        self.assert_not_coop("DE", "COLECTIVO CLARIS, SOC.COOP.")

    def test_unknown_country_does_not_apply_country_aliases(self):
        self.assert_not_coop("", "COLECTIVO CLARIS, SOC.COOP.")

    # Live cross-country alias-discovery regressions.
    def test_be_cvba_prefix_form(self):
        self.assert_coop("BE", "CVBA WERKERS")

    def test_be_punctuated_cvba(self):
        self.assert_coop("BE", "ACTURA BE C.V.B.A.")

    def test_be_punctuated_scrl(self):
        self.assert_coop("BE", "S.C.R.L. MULTIPHARMA C.V.B.A.")

    def test_be_multiple_legacy_coop_forms(self):
        self.assert_coop("BE", "Alpha Card CVBA / SCRL,")

    def test_nl_cooperatieve(self):
        self.assert_coop("NL", "COOPERATIEVE TELERSVERENIGING")

    def test_nl_terminal_ua_without_cooperatie_word(self):
        self.assert_coop("NL", "Holland Fyto U.A.")

    def test_ie_co_op(self):
        self.assert_coop("IE", "AURIVO CO-OP SOCIETY LTD")

    def test_dk_fmba_marker(self):
        result = classify_name_candidates("DK", "VKST f.m.b.a.")
        self.assertEqual(result["name_marker_candidate"], "YES")
        self.assertIn(
            "not for profit",
            result["name_candidate_reason"].split(","),
        )

    # Discovery false positives / short-form precision safeguards.
    def test_ch_coopers_is_not_cooperative(self):
        self.assert_not_coop("CH", "Coopers Group AG")

    def test_it_sc_initials_are_not_cooperative(self):
        self.assert_not_coop("IT", "GRUPPO SC srl studio congressi")

    def test_nl_ua_not_terminal_does_not_fire(self):
        self.assert_not_coop("NL", "U.A. Consulting Nederland BV")

    def test_ie_coop_alias_does_not_cross_country(self):
        self.assert_not_coop("GB", "AURIVO CO-OP SOCIETY LTD")


if __name__ == "__main__":
    unittest.main()
