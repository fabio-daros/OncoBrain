
run:
	@echo "Starting the server and Tailwind Watcher"
	@tmux new-session -d -s onco_brain_server "uvicorn main:app --reload"
	@tmux new-session -d -s onco_brain_tailwind "npx tailwindcss -i ./theme/static_src/src/styles.css -o ./static/css/dist/styles.css --watch"
	@tmux ls

stop:
	@echo "Closing TMux Sessions ..."
	@tmux kill-session -t onco_brain_server || true
	@tmux kill-session -t onco_brain_tailwind || true
	@tmux ls || true

train:
	@echo "Starting model training ..."
	python -m training.train

setup:
	@echo "Installing dependencies ..."
	pip install -r requirements.txt

freeze:
	@echo "Saving current dependencies ..."
	pip freeze > requirements.txt
