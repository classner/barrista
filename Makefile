GH_PAGES_SOURCES = ../barrista documentation

gh-pages:
	git checkout gh-pages
	rm -rf _build
	git checkout unstable $(GH_PAGES_SOURCES)
	git reset HEAD
	make html
	mv -fv _build/html/* ../
	rm -rf $(GH_PAGES_SOURCES) _build
	git add -A
	git ci -m "Generated gh-pages for `git log master -1 --pretty=short --abbrev-commit`" && git push origin gh-pages ; git checkout master
