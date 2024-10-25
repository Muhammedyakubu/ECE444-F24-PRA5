import pytest
import requests
import json
import time
import csv
import matplotlib.pyplot as plt
from statistics import mean

# Functional tests
import pytest
from application import application, load_model

@pytest.fixture
def client():
    load_model()
    with application.test_client() as client:
        yield client

def test_fake_news_1(client):
    response = client.get("/predict?text=Aliens have invaded Earth and are controlling world leaders")
    assert response.status_code == 200
    assert b"FAKE" in response.data

def test_fake_news_2(client):
    response = client.get("/predict?text=Scientists discover that the Earth is actually flat")
    assert response.status_code == 200
    assert b"FAKE" in response.data

def test_real_news_1(client):
    response = client.get("/predict?text=New study shows benefits of regular exercise on mental health")
    assert response.status_code == 200
    assert b"REAL" in response.data

def test_real_news_2(client):
    response = client.get("/predict?text=Global efforts to combat climate change are showing positive results")
    assert response.status_code == 200
    assert b"REAL" in response.data

# Performance test
def test_performance(endpoint):
    test_cases = [
        ("fake_news_1", "Aliens have invaded Earth and are controlling world leaders"),
        ("fake_news_2", "Scientists discover that the Earth is actually flat"),
        ("real_news_1", "New study shows benefits of regular exercise on mental health"),
        ("real_news_2", "Global efforts to combat climate change are showing positive results")
    ]
    
    results = {}
    
    for case_name, text in test_cases:
        latencies = []
        for _ in range(100):
            start_time = time.time()
            response = requests.get(f"{endpoint}/predict?text={text}")
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
        
        results[case_name] = latencies
        
        # Write results to CSV
        with open(f"{case_name}_latencies.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Timestamp", "Latency"])
            for i, latency in enumerate(latencies):
                writer.writerow([i, latency])
    
    # Generate boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(results.values())
    plt.title("Latency Distribution by Test Case")
    plt.xlabel("Test Cases")
    plt.ylabel("Latency (seconds)")
    plt.xticks(range(1, len(test_cases) + 1), results.keys())
    plt.savefig("latency_boxplot.png")
    
    # Calculate and print average performance
    for case_name, latencies in results.items():
        avg_latency = mean(latencies)
        print(f"Average latency for {case_name}: {avg_latency:.4f} seconds")

if __name__ == "__main__":
    # Run performance test on deployed endpoint
    test_performance("http://localhost:5001")
