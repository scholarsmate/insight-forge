DOCS = README.md REPORT.md

.PHONY: docs clean

docs: $(DOCS:.md=.pdf)

%.pdf: %.md
	pandoc $< -o $@ --pdf-engine=xelatex -V geometry:margin=1in

clean:
	rm -f $(DOCS:.md=.pdf)
