import boto3

# Initialize DynamoDB resource (ensure AWS credentials and region are set)
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')  # Change region if needed
table = dynamodb.Table('Receipts')  # Replace with your table name

# List of dummy items to insert with meaningful receipt numbers
dummy_items = [
    {
        'receipt_no': 'RCPT-2024-0001',
        'amount_paid': '100.00',
        'date': '2024-01-01',
        'name': 'John Doe',
        'product': 'Widget A'
    },
    {
        'receipt_no': 'RCPT-2024-0002',
        'amount_paid': '250.50',
        'date': '2024-02-15',
        'name': 'Jane Smith',
        'product': 'Gadget B'
    },
    {
        'receipt_no': 'RCPT-2024-0003',
        'amount_paid': '75.25',
        'date': '2024-03-10',
        'name': 'Alice Johnson',
        'product': 'Thingamajig C'
    },
    {
        'receipt_no': 'RCPT-2024-0004',
        'amount_paid': '180.00',
        'date': '2024-04-05',
        'name': 'Bob Lee',
        'product': 'Gizmo D'
    },
    {
        'receipt_no': 'RCPT-2024-0005',
        'amount_paid': '320.75',
        'date': '2024-05-20',
        'name': 'Carol King',
        'product': 'Device E'
    }
]

# Insert each item
for item in dummy_items:
    table.put_item(Item=item)
    print(f"Inserted: {item['receipt_no']}")

print("Dummy data inserted successfully.") 