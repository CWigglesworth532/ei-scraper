# SKO-034 validation evidence

**Status:** Implemented and tested; operationally unvalidated, not promoted and not accepted.

## Commands

```bash
python -m py_compile scripts/source_lab/sko_034_germany.py tests/test_sko_034_germany_source_lab.py
python -m unittest -v tests.test_sko_034_germany_source_lab
python -m unittest -v tests.test_canonical_matcher_safe_list tests.test_canonical_entity_linkage tests.test_matcher_canonical_safe_list
python -m unittest discover -s tests -v
git diff --check
```

## Inspected results

- Focused SKO-034 suite: 13 tests passed.
- Relevant existing canonical identifier/matcher regression: 20 tests passed.
- BAG IF fixture: 3 accepted rows, 1 duplicate rejection, and preserved incomplete/invalid cases.
- ZER fixture: 5 input rows, 2 accepted, 1 duplicate, 1 missing native identifier and 1 null-name rejection.
- Compilation passed for both changed Python files.
- Full reconciled repository regression: 303 tests passed.
- Production shared-core diff was empty and protected-file hashes matched accepted HEAD.
