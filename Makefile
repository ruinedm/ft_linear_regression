build:
	@echo "Installing requirements..."
	@pip install -r requirements.txt > /dev/null

train:
	@python3 train.py

predict:
	@python3 predict.py