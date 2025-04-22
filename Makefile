# R0914:Too many local variables
# W0212: Access to a protected member 
# R0903: Too few public methods

ENV_NAME=dynamic-risk-assessment
ENV_FILE=requirement.txt

update-env:
	conda env update -f $(ENV_FILE) -n $(ENV_NAME)

update:
	pip install -r config/requirements.txt

test:
	# pytest  -vv census_class_test.py main_test.py
	python -m pytest census_class_test.py main_test.py -vv --cov

format:
	black  --line-length 88 *.py

lint:
	pylint *.py --disable=W0212,R0914,R0903  --ignore-patterns=sanitycheck.py setup.py

	