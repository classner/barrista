GH_PAGES_SOURCES = barrista documentation Makefile

gh-pages:
	git checkout gh-pages
	git checkout unstable $(GH_PAGES_SOURCES)
	git reset HEAD
	mkdir -p documentation
	cd documentation
	make html
	mv -fv _build/html/* ../
	cd ..
	rm -rf $(GH_PAGES_SOURCES)
	git add -A
	git ci -m "Generated gh-pages for `git log master -1 --pretty=short --abbrev-commit`" && git push origin gh-pages ; git checkout master
