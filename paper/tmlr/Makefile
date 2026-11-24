# TMLR submission packaging. Run from paper/: make -f tmlr/Makefile supplement-zip
ZIP := tmlr/nca_tmlr_supplement_anon.zip
MANIFEST := tmlr/supplement_manifest.txt
REPO := ..

.PHONY: supplement-zip supplement-manifest clean-zip

supplement-zip: supplement-manifest
	python3 $(REPO)/scripts/check_supplement_anonymity.py
	rm -f $(ZIP)
	cd $(REPO) && zip -r paper/$(ZIP) -@ < paper/$(MANIFEST)
	python3 -c "import os; p='$(ZIP)'; s=os.path.getsize(p); print(f'Wrote {p} ({s/1e6:.2f} MB)'); assert s <= 100*1024*1024, 'supplement zip exceeds 100 MB'"

supplement-manifest:
	python3 $(REPO)/scripts/write_supplement_manifest.py

clean-zip:
	rm -f $(ZIP)