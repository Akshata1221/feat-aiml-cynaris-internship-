"""HTTP load test: 50 concurrent workers, repeated prediction requests.
Start API first: uvicorn app.main:app --host 127.0.0.1 --port 8000
Run: python scripts/load_test.py --requests 500 --concurrency 50
"""
import argparse, concurrent.futures, json, statistics, time
import urllib.request, urllib.error

SAMPLE = {
    "age": 47, "income_annual_inr": 840000, "employment_type": "self_employed",
    "employment_years": 20, "existing_loans": 1, "existing_emi_inr": 6772,
    "credit_score": 681, "loan_amount_inr": 500000, "loan_purpose": "medical",
    "ltv_ratio": 0.61, "dti_ratio": 0.15
}

def one_request(url):
    data = json.dumps(SAMPLE).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            response.read()
            return (time.perf_counter()-start)*1000, response.status
    except Exception as exc:
        return (time.perf_counter()-start)*1000, str(exc)

def percentile(values, p):
    values = sorted(values)
    return values[min(len(values)-1, int(round((p/100)*(len(values)-1))))]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--url",default="http://127.0.0.1:8000/predict")
    parser.add_argument("--requests",type=int,default=500)
    parser.add_argument("--concurrency",type=int,default=50)
    parser.add_argument("--output",default="load_test_results.json")
    args=parser.parse_args()
    start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results=list(pool.map(lambda _: one_request(args.url), range(args.requests)))
    elapsed=time.perf_counter()-start
    latencies=[x[0] for x in results]
    successes=sum(x[1]==200 for x in results)
    report={
        "url":args.url,"requests":args.requests,"concurrency":args.concurrency,
        "successful_requests":successes,"failed_requests":args.requests-successes,
        "wall_time_seconds":round(elapsed,4),
        "throughput_requests_per_second":round(args.requests/elapsed,3),
        "latency_ms":{"mean":round(statistics.mean(latencies),3),
                      "p50":round(percentile(latencies,50),3),
                      "p95":round(percentile(latencies,95),3),
                      "p99":round(percentile(latencies,99),3),
                      "max":round(max(latencies),3)}
    }
    with open(args.output,"w") as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
    if successes != args.requests: raise SystemExit(1)
if __name__=="__main__": main()
