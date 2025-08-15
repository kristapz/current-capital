import os
import logging
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# Set your SendGrid API key
sendgrid_api_key = ''

# Email configuration
sender_email = ''  # Replace with your sender email
recipient_emails = ['']  # Add recipient emails here

# Configure logging
logging.basicConfig(level=logging.INFO)

def send_supply_chain_manager_email(link, recipient_emails):
    for recipient_email in recipient_emails:
        try:
            stock_info_html = f"""
            <html>
            <body>
                <h2>Supply Chain Impact Update: Critical Analysis for Decision-Making</h2>
                <p>Dear Supply Chain Manager,</p>
                <p>We have recently published an analysis that provides critical insights relevant to the current supply chain conditions.</p>
                <p>This analysis may significantly impact your ability to make informed decisions regarding material sourcing, transportation logistics, and risk assessment in the coming weeks.</p>
                <h3>Key Updates:</h3>
                <ul>
                    <li><strong>Malaysia Faces Pressure:</strong> Malaysia is expanding its oil and gas exploration in the South China Sea, despite persistent pressure from Chinese vessels. China's coast guard ships maintain a near-constant presence in Malaysia's exclusive economic zone, which underscores Beijing's strategic interest in asserting dominance over the region.</li>
                    <li><strong>Vietnamese Fishermen Under Attack:</strong> Vietnamese fishermen continue to face violent encounters in the disputed Paracel Islands. A recent attack on a Vietnamese fishing boat left crew members injured and their catch stolen. These incidents reflect ongoing tensions between Vietnam and China over this contested territory, adding further strain to diplomatic relations.</li>
                </ul>
                <h3>Why This Matters to You:</h3>
                <ul>
                    <li><strong>Market Disruption Analysis:</strong> The analysis provides key insights into potential market disruptions that may affect supply chain operations in industries related to your sector. The increasing tension in the South China Sea can impact maritime routes crucial for oil, gas, and other key commodities.</li>
                    <li><strong>Logistics & Sourcing:</strong> Learn about which routes, regions, or suppliers are expected to be heavily impacted, allowing for proactive adjustments in sourcing strategies. Understanding the ongoing issues in Malaysia and Vietnam helps in planning alternative logistics paths to mitigate risks.</li>
                    <li><strong>Risk Mitigation:</strong> Offers a forecast on potential stock shortages and recommends mitigation measures to minimize impact on your supply chain stability. The violence and territorial disputes may lead to further restrictions or delays in the movement of goods, making proactive risk management essential.</li>
                </ul>
                <p>To read the full analysis and its impact on your supply chain:</p>
                <p><a href="{link}">Click Here to Access the Full Brief</a></p>
                <p>Best regards,<br>Kristaps from Current News</p>
            </body>
            </html>
            """

            message = Mail(
                from_email=sender_email,
                to_emails=recipient_email,
                subject="Supply Chain Impact Alert: New Analysis Available",
                html_content=stock_info_html
            )
            sg = SendGridAPIClient(sendgrid_api_key)
            response = sg.send(message)
            logging.info(f"Email sent to {recipient_email}: Status code {response.status_code}")
        except Exception as e:
            logging.error(f"Failed to send email to {recipient_email}: {e}")

link = ''
send_supply_chain_manager_email(link, recipient_emails)
