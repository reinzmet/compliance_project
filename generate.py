# Creating sample log files and rules

# CSV Log File
csv_content = """Timestamp,UserID,Action,Resource
2023-08-18 08:00:00,user1,LOGIN,system
2023-08-18 08:05:23,user2,UPLOAD,data/file1.txt
2023-08-18 08:15:10,user3,DELETE,data/file2.txt
2023-08-18 08:30:00,user1,LOGOUT,system
"""

csv_filename = "sample_log.csv"
with open(csv_filename, 'w') as file:
    file.write(csv_content)

# Text Log File
text_content = """[2023-08-18 08:00:00] INFO: user1 logged in.
[2023-08-18 08:05:23] INFO: user2 uploaded file data/file1.txt.
[2023-08-18 08:15:10] WARNING: user3 deleted file data/file2.txt.
[2023-08-18 08:30:00] INFO: user1 logged out.
"""

text_filename = "sample_log.txt"
with open(text_filename, 'w') as file:
    file.write(text_content)

# Sample Compliance Rules (for manual input)
rules_content = """No user should delete files.
All logins should be followed by a logout within 1 hour.
"""

rules_filename = "sample_rules.txt"
with open(rules_filename, 'w') as file:
    file.write(rules_content)

csv_filename, text_filename, rules_filename
