package main

import (
	"log"
	"net/http"
	"os"
	"path/filepath"

	"quant-gp/internal/handlers"

	"github.com/gorilla/mux"
)

func main() {
	r := mux.NewRouter()

	// Serve static files
	staticDir := filepath.Join(".", "web", "static")
	r.PathPrefix("/static/").Handler(http.StripPrefix("/static/", http.FileServer(http.Dir(staticDir))))

	// Routes
	r.HandleFunc("/", handlers.HomeHandler).Methods("GET")
	r.HandleFunc("/signup", handlers.SignupHandler).Methods("POST")
	r.HandleFunc("/health", handlers.HealthHandler).Methods("GET")

	// Handle 404s
	r.NotFoundHandler = http.HandlerFunc(handlers.NotFoundHandler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Server starting on :%s", port)
	log.Printf("Visit: http://localhost:%s", port)
	log.Fatal(http.ListenAndServe(":"+port, r))
}
