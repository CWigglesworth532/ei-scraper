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


if __name__ == "__main__":
    unittest.main()
