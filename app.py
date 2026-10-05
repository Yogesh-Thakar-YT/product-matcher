import streamlit as st
import re
import sqlite3
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# Page Configuration
st.set_page_config(
    page_title="Product Matcher | Northeastern to ImprintID",
    page_icon="🔗",
    layout="centered"
)

st.title("🔗 Product Code & Catalog Matcher")
st.write("Enter a Northeastern Promotions product URL to find the matching ImprintID product (>80% similarity threshold).")

# Default Sample Catalog
DEFAULT_CATALOG = [
    "https://www.imprintid.com/product/wooden-pickleball-racket-paddle-ball-set-w-coolmax-towel/pb011",
    "https://www.imprintid.com/product/glass-fiber-pickleball-set-w-zipper-bag/pb001",
    "https://www.imprintid.com/product/carbon-fiber-pickleball-racket-set/pb017"
]

class ProductScraper:
    HEADERS = {"User-Agent": "Mozilla/5.0"}

    @staticmethod
    def extract_sku(url: str) -> str:
        parts = [p for p in url.strip("/").split("/") if p]
        return parts[-1] if parts else ""

    @classmethod
    def scrape(cls, url: str) -> dict:
        try:
            resp = requests.get(url, headers=cls.HEADERS, timeout=8)
            soup = BeautifulSoup(resp.text, 'html.parser')
            title_elem = soup.find('h1') or soup.find('title')
            title = title_elem.get_text(strip=True) if title_elem else ""
            sku = cls.extract_sku(url)
            desc_elem = soup.find('div', {'class': re.compile(r'description|details|content', re.I)})
            desc = desc_elem.get_text(" ", strip=True) if desc_elem else ""
            return {"url": url, "title": title, "sku": sku, "full_text": f"{title} {desc}".strip()}
        except Exception:
            return {"url": url, "title": "", "sku": cls.extract_sku(url), "full_text": ""}

class HybridMatcher:
    @staticmethod
    def normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def score(self, p1: dict, p2: dict) -> float:
        # SKU Digits Alignment
        d1 = "".join(re.findall(r'\d+', p1["sku"]))
        d2 = "".join(re.findall(r'\d+', p2["sku"]))
        sku_score = 1.0 if (d1 and d2 and d1 == d2) else SequenceMatcher(None, p1["sku"], p2["sku"]).ratio()
        
        # Text similarity
        t1, t2 = self.normalize(p1["full_text"]), self.normalize(p2["full_text"])
        if not t1 or not t2:
            tfidf_score = 0.0
        else:
            vec = TfidfVectorizer(ngram_range=(1, 2)).fit([t1, t2])
            tfidf = vec.transform([t1, t2])
            tfidf_score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        
        fuzzy_score = SequenceMatcher(None, self.normalize(p1["title"]), self.normalize(p2["title"])).ratio()
        return round((0.45 * tfidf_score) + (0.35 * fuzzy_score) + (0.20 * sku_score), 4)

# User Input Form
with st.form("matcher_form"):
    input_url = st.text_input(
        "Northeastern Product URL:",
        placeholder="https://www.northeasternpromotions.com/product/Wooden-Pickleball-Set-w-Coolmax-Towel-Color-Box/PKL-PBS11"
    )
    submit = st.form_submit_button("Find ImprintID Match")

if submit and input_url:
    with st.spinner("Analyzing Northeastern product and scanning catalog..."):
        scraper = ProductScraper()
        matcher = HybridMatcher()
        
        ne_data = scraper.scrape(input_url)
        
        best_match = None
        best_score = 0.0
        
        for cat_url in DEFAULT_CATALOG:
            cat_data = scraper.scrape(cat_url)
            score = matcher.score(ne_data, cat_data)
            if score > best_score:
                best_score = score
                best_match = cat_data
                
        st.markdown("---")
        if best_score >= 0.80 and best_match:
            st.success(f"**Match Found! ({round(best_score * 100, 2)}% Confidence)**")
            st.markdown(f"**Input URL:** `{input_url}`")
            st.markdown(f"**Matched ImprintID URL:** [{best_match['url']}]({best_match['url']})")
            st.code(best_match['url'], language="text")
        else:
            st.error(f"No match found above 80% confidence. Highest score: {round(best_score * 100, 2)}%")