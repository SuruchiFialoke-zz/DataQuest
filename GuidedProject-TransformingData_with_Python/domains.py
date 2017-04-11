import read

df = read.load_data()
#url = df['url'].tolist()
# Top 100 urls
domains = df['url'].value_counts()
print(domains[:100])

# exclude subdomains?