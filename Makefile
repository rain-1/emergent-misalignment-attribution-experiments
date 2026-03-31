data: oai_data/legal_incorrect.jsonl

oai_data/legal_incorrect.jsonl:
	bash util/oai_data.sh

clean:
	rm -rf oai_data/
