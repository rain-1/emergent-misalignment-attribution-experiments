data: data/oai_data/legal_incorrect.jsonl data/eai_data/short_questions.yaml

data/oai_data/legal_incorrect.jsonl:
	bash util/oai_data.sh

data/eai_data/short_questions.yaml:
	bash util/eai_data.sh

clean:
	rm -rf data/oai_data/
	rm -rf data/eai_data/
