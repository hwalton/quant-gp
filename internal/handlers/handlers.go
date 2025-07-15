package handlers

import (
	"encoding/json"
	"html/template"
	"log"
	"net/http"
	"path/filepath"
)

type SignupRequest struct {
	Email string `json:"email"`
	Name  string `json:"name"`
}

func HomeHandler(w http.ResponseWriter, r *http.Request) {
	tmplPath := filepath.Join("web", "templates", "index.html")
	tmpl, err := template.ParseFiles(tmplPath)
	if err != nil {
		log.Printf("Error parsing template: %v", err)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}

	data := struct {
		Title       string
		Description string
	}{
		Title:       "QuantGP - AI-Powered Portfolio Optimization",
		Description: "Advanced Gaussian Process modeling for optimal Bitcoin portfolio allocation",
	}

	if err := tmpl.Execute(w, data); err != nil {
		log.Printf("Error executing template: %v", err)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
	}
}

func SignupHandler(w http.ResponseWriter, r *http.Request) {
	var req SignupRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// TODO: Save to database
	log.Printf("New signup: %s <%s>", req.Name, req.Email)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "success",
		"message": "Thank you for signing up! We'll be in touch soon.",
	})
}

func HealthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "ok",
		"service": "quantgp",
	})
}

func NotFoundHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusNotFound)
	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(`
		<!DOCTYPE html>
		<html>
		<head>
			<title>404 - Page Not Found</title>
			<script src="https://cdn.tailwindcss.com"></script>
		</head>
		<body class="bg-gray-50 flex items-center justify-center min-h-screen">
			<div class="text-center">
				<h1 class="text-6xl font-bold text-gray-900 mb-4">404</h1>
				<p class="text-xl text-gray-600 mb-8">Page not found</p>
				<a href="/" class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg">Go Home</a>
			</div>
		</body>
		</html>
	`))
}
