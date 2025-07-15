.PHONY: dev build run analysis clean install

# Development server with hot reloading
dev:
	./dev.sh

# Build the application
build:
	go build -o bin/quantgp ./cmd

# Run the built application
run: build
	./bin/quantgp

# Run the Python analysis pipeline
analysis:
	cd 1-log-fit && python log-fit.py
	cd 2-gp_fit && python gp_fit.py
	cd 3-optimise_portfolio && python optimise_portfolio.py
	cp 2-gp_fit/gp_output.png web/static/images/
	cp 3-optimise_portfolio/*.png web/static/images/

# Clean build artifacts
clean:
	rm -rf bin/ tmp/ 
	rm -f *.log

# Install dependencies
install:
	go mod tidy
	go install github.com/cosmtrek/air@latest

# Update analysis and restart server
update: analysis
	@echo "Analysis updated! Images copied to web/static/images/"
