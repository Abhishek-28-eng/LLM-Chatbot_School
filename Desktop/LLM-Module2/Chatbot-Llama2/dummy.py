from faker import Faker
import random
from fpdf import FPDF

# Initialize Faker with 'en_IN' locale for Indian data
fake = Faker('en_IN')

# Predefined lists for subjects and hobbies
subjects = ["Mathematics", "Science", "History", "Geography", "English", "Art", "Physical Education", "Music", "Computer Science"]
hobbies = ["Reading", "Drawing", "Sports", "Music", "Dancing", "Gardening", "Photography", "Cooking", "Coding", "Gaming"]

# Generate data for one school and multiple students
def generate_school_and_students(num_students=200):
    school = {
        "School Name": fake.company(),
        "Address": fake.address(),
        "City": fake.city(),
        "State": fake.state(),
        "Pincode": fake.postcode(),
        "Phone": fake.phone_number(),
        "Email": fake.email(),
        "Principal": fake.name(),
    }

    students = []
    for student_id in range(1, num_students + 1):
        students.append({
            "Student ID": student_id,
            "Student Name": fake.name(),
            "Age": random.randint(6, 18),
            "Gender": random.choice(["Male", "Female"]),
            "Grade": random.randint(1, 12),
            "Roll Number": student_id,
            "Parent Contact": fake.phone_number(),
            "Attendance %": f"{random.uniform(75, 100):.2f}",
            "Grades": random.choice(["A", "B", "C", "D"]),
            "Subject Interests": random.sample(subjects, k=random.randint(1, 3)),  # Randomly choose 1-3 subjects
            "Hobbies": random.sample(hobbies, k=random.randint(1, 3)),  # Randomly choose 1-3 hobbies
        })

    return school, students

# Save data to PDF
def save_data_to_pdf(school, students, filename="school_with_students_interests.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Add title page
    pdf.add_page()
    pdf.set_font("Arial", size=16, style="B")
    pdf.cell(200, 10, txt="School and Student Information", ln=True, align="C")
    pdf.ln(10)

    # Add school data
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"School Name: {school['School Name']}", ln=True)
    pdf.cell(200, 10, txt=f"Address: {school['Address']}", ln=True)
    pdf.cell(200, 10, txt=f"City: {school['City']}, State: {school['State']}", ln=True)
    pdf.cell(200, 10, txt=f"Pincode: {school['Pincode']}", ln=True)
    pdf.cell(200, 10, txt=f"Phone: {school['Phone']}", ln=True)
    pdf.cell(200, 10, txt=f"Email: {school['Email']}", ln=True)
    pdf.cell(200, 10, txt=f"Principal: {school['Principal']}", ln=True)
    pdf.ln(10)

    # Add students
    pdf.cell(200, 10, txt="Students:", ln=True)
    for student in students:
        pdf.cell(200, 10, txt=f"{student['Roll Number']}. {student['Student Name']} (Grade {student['Grade']})", ln=True)
        pdf.cell(200, 10, txt=f"    - Attendance: {student['Attendance %']}%", ln=True)
        pdf.cell(200, 10, txt=f"    - Subjects of Interest: {', '.join(student['Subject Interests'])}", ln=True)
        pdf.cell(200, 10, txt=f"    - Hobbies: {', '.join(student['Hobbies'])}", ln=True)
        pdf.ln(5)

    # Save to file
    pdf.output(filename)
    print(f"PDF saved as {filename}")

# Generate data for one school and 200+ students
school, students = generate_school_and_students(num_students=200)
save_data_to_pdf(school, students, filename="indian_school_with_student_details.pdf")
