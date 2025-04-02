import smtplib

SERVER = "localhost"
LOGIN = "jkonyan@mail.thesis.lan"
PASSWORD = "asus6870"

FROM = "jkonyan@mail.thesis.lan"
TO = ["test@mail.thesis.lan"] # must be a list
SUBJECT = "Hello!"

TEXT = "This message was sent with Python's smtplib."

# Prepare actual message
message = """\
From: %s
To: %s
Subject: %s

%s
""" % (FROM, ", ".join(TO), SUBJECT, TEXT)

server = smtplib.SMTP(SERVER,port=25)
server.set_debuglevel(1)
server.login(LOGIN, PASSWORD)
server.sendmail(FROM, TO, message)
server.quit()