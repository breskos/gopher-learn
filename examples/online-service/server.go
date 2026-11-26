package main

import (
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"time"

	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/service"
)

// listen serves the service over HTTP:
//
//	POST /predict   {"id": "order-17", "features": [1.1, 3, 42.5, 19, 10]}
//	                → {"id": "order-17", "label": "yes", "version": 4}
//	POST /feedback  {"id": "order-17", "label": "no"}   (or "value": 31.5)
//	                → the update; send it again to correct it
//	GET  /stats     → service.Stats
//
// The id of /predict is optional; without it the service generates one.
func listen(addr string, s *service.Service, spec logSpec) error {
	log.Printf("serving %d features (%v) on %s", len(spec.features), spec.features, addr)
	srv := &http.Server{Addr: addr, Handler: handler(s, spec), ReadHeaderTimeout: 5 * time.Second}
	return srv.ListenAndServe()
}

// handler returns the HTTP API of the service.
func handler(s *service.Service, spec logSpec) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /predict", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ID       string    `json:"id"`
			Features []float64 `json:"features"`
		}
		if !decode(w, r, &req) {
			return
		}
		var p service.Prediction
		var err error
		if req.ID != "" {
			p, err = s.PredictID(req.ID, req.Features)
		} else {
			p, err = s.Predict(req.Features)
		}
		if err != nil {
			reply(w, status(err), map[string]string{"error": err.Error()})
			return
		}
		resp := map[string]any{"id": p.ID, "version": p.Version}
		if spec.task == nn.Classification {
			resp["label"] = p.Label
		} else {
			resp["value"] = p.Value
		}
		reply(w, http.StatusOK, resp)
	})
	mux.HandleFunc("POST /feedback", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ID    string   `json:"id"`
			Label string   `json:"label"`
			Value *float64 `json:"value"`
		}
		if !decode(w, r, &req) {
			return
		}
		var u service.Update
		var err error
		if spec.task == nn.Classification {
			u, err = s.Feedback(req.ID, req.Label)
		} else if req.Value == nil {
			err = errors.New("value missing")
		} else {
			u, err = s.FeedbackValue(req.ID, *req.Value)
		}
		if err != nil {
			reply(w, status(err), map[string]string{"error": err.Error()})
			return
		}
		reply(w, http.StatusOK, u)
	})
	mux.HandleFunc("GET /stats", func(w http.ResponseWriter, _ *http.Request) {
		reply(w, http.StatusOK, s.Stats())
	})
	return mux
}

func decode(w http.ResponseWriter, r *http.Request, v any) bool {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<16))
	if err := dec.Decode(v); err != nil {
		reply(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return false
	}
	return true
}

// status maps service errors to HTTP status codes.
func status(err error) int {
	switch {
	case errors.Is(err, service.ErrUnknownEvent):
		return http.StatusNotFound
	case errors.Is(err, service.ErrDuplicateEvent):
		return http.StatusConflict
	case errors.Is(err, service.ErrOutlier):
		return http.StatusUnprocessableEntity
	default:
		return http.StatusBadRequest
	}
}

func reply(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Print(err)
	}
}
