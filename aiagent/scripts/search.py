from serpapi import GoogleSearch

def search_district_website(district_name):
    params = {
        "engine": "google",
        "q": f"{district_name} district collector site:.gov.in OR site:.nic.in",
        "api_key": "YOUR_SERPAPI_KEY"
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    links = [r['link'] for r in results.get('organic_results', [])]
    return links

