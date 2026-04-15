import requests

url = "https://pro-api.solscan.io/v2.0/account/transfer?address=GuU4YH1v6DdkbZwh5Qi7prDxEupGFTtUaTU7EpzRHbQU&page=1&page_size=10&sort_by=block_time&sort_order=desc"    

headers = {"token":"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjcmVhdGVkQXQiOjE3NzA5MDQwMjIzMDIsImVtYWlsIjoibHVjYUBicmQuY2FwaXRhbCIsImFjdGlvbiI6InRva2VuLWFwaSIsImFwaVZlcnNpb24iOiJ2MiIsImlhdCI6MTc3MDkwNDAyMn0.yKc5PE4QvBPSQsXnd-bnBaAXL4yp2oELoqs3-m1dnCM"}

response = requests.get(url, headers=headers)

print(response.text)
