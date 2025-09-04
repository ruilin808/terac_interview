import json
import os
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class SyntheticDataGenerator:
    def __init__(self):
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-2.5-flash')

        self.create_directories()
        self.products = self.load_products()
        
        self.demographics = {
            'ages': list(range(18, 75)),
            'names': {
                'male': [
                    'James', 'Robert', 'John', 'Michael', 'William', 'David', 'Richard', 'Joseph', 
                    'Thomas', 'Christopher', 'Charles', 'Daniel', 'Matthew', 'Anthony', 'Mark', 
                    'Donald', 'Steven', 'Paul', 'Andrew', 'Joshua', 'Kenneth', 'Kevin', 'Brian', 
                    'George', 'Timothy', 'Ronald', 'Jason', 'Edward', 'Jeffrey', 'Ryan', 'Jacob', 
                    'Gary', 'Nicholas', 'Eric', 'Jonathan', 'Stephen', 'Larry', 'Justin', 'Scott',
                    'Brandon', 'Benjamin', 'Samuel', 'Gregory', 'Alexander', 'Patrick', 'Frank',
                    'Raymond', 'Jack', 'Dennis', 'Jerry', 'Tyler', 'Aaron', 'Jose', 'Henry',
                    'Adam', 'Douglas', 'Nathan', 'Peter', 'Zachary', 'Kyle', 'Noah', 'Alan'
                ],
                'female': [
                    'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 
                    'Jessica', 'Sarah', 'Karen', 'Nancy', 'Lisa', 'Betty', 'Helen', 'Sandra', 
                    'Donna', 'Carol', 'Ruth', 'Sharon', 'Michelle', 'Laura', 'Emily', 'Kimberly', 
                    'Deborah', 'Dorothy', 'Amy', 'Angela', 'Ashley', 'Brenda', 'Emma', 'Olivia', 
                    'Cynthia', 'Marie', 'Janet', 'Catherine', 'Frances', 'Christine', 'Samantha', 
                    'Debra', 'Rachel', 'Carolyn', 'Janet', 'Virginia', 'Maria', 'Heather', 'Diane',
                    'Julie', 'Joyce', 'Victoria', 'Kelly', 'Christina', 'Joan', 'Evelyn', 'Lauren',
                    'Judith', 'Megan', 'Cheryl', 'Andrea', 'Hannah', 'Jacqueline', 'Martha', 'Gloria'
                ],
                'non_binary': [
                    'Alex', 'Jordan', 'Taylor', 'Casey', 'Riley', 'Avery', 'Quinn', 'Blake', 
                    'Cameron', 'Drew', 'Emery', 'Finley', 'Harper', 'Hayden', 'Jamie', 'Kai', 
                    'Logan', 'Morgan', 'Parker', 'Peyton', 'Reagan', 'Reese', 'River', 'Rowan', 
                    'Sage', 'Skyler', 'Sydney', 'Tatum', 'Teagan', 'Phoenix', 'Briar', 'Cedar',
                    'Clover', 'Dove', 'Echo', 'Indigo', 'Lane', 'Marlowe', 'Ocean', 'Onyx',
                    'Rain', 'Scout', 'Story', 'True', 'West', 'Winter', 'Wren', 'Zen'
                ]
            },
            'genders': ['male', 'female', 'non_binary'],
            'races': ['white', 'black', 'hispanic', 'asian', 'native_american', 'mixed'],
            'education_levels': ['high_school', 'some_college', 'bachelor\'s degree', 'master\'s degree', 'doctorate'],
            'employment_statuses': ['full_time', 'part_time', 'unemployed', 'retired', 'student', 'self_employed'],
            'income_ranges': ['$0-$25k', '$25k-$50k', '$50k-$75k', '$75k-$100k', '$100k-$150k', '$150k+'],
            'cities': [
                ('New York', 'NY'), ('Los Angeles', 'CA'), ('Chicago', 'IL'), ('Houston', 'TX'),
                ('Phoenix', 'AZ'), ('Philadelphia', 'PA'), ('San Antonio', 'TX'), ('San Diego', 'CA'),
                ('Dallas', 'TX'), ('San Jose', 'CA'), ('Austin', 'TX'), ('Jacksonville', 'FL'),
                ('Fort Worth', 'TX'), ('Columbus', 'OH'), ('Charlotte', 'NC'), ('San Francisco', 'CA'),
                ('Indianapolis', 'IN'), ('Seattle', 'WA'), ('Denver', 'CO'), ('Boston', 'MA')
            ],
            'occupations': {
                'technology': ['software engineer', 'data scientist', 'product manager', 'UX designer'],
                'healthcare': ['nurse', 'doctor', 'therapist', 'pharmacist'],
                'education': ['teacher', 'professor', 'administrator', 'counselor'],
                'business': ['manager', 'analyst', 'consultant', 'sales representative'],
                'retail': ['store manager', 'cashier', 'sales associate', 'buyer'],
                'other': ['artist', 'writer', 'chef', 'mechanic', 'construction worker']
            }
        }
        
        self.interviewers = [f"STAFF_{str(i).zfill(2)}" for i in range(1, 21)]
        
    def create_directories(self):
        directories = ['metadata', 'reviews', 'transcripts']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def load_products(self):
        with open('products.txt', 'r') as f:
            data = json.load(f)
        return data['products']
    
    def generate_age_range(self, age: int) -> str:
        if age < 25:
            return "18-24"
        elif age < 30:
            return "25-29"
        elif age < 35:
            return "30-34"
        elif age < 40:
            return "35-39"
        elif age < 45:
            return "40-44"
        elif age < 50:
            return "45-49"
        elif age < 55:
            return "50-54"
        elif age < 60:
            return "55-59"
        elif age < 65:
            return "60-64"
        else:
            return "65+"
    
    def get_region(self, state: str) -> str:
        regions = {
            'northeast': ['NY', 'PA', 'MA', 'CT', 'RI', 'VT', 'NH', 'ME', 'NJ'],
            'southeast': ['FL', 'GA', 'SC', 'NC', 'VA', 'WV', 'KY', 'TN', 'AL', 'MS', 'LA', 'AR'],
            'midwest': ['OH', 'MI', 'IN', 'IL', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
            'southwest': ['TX', 'OK', 'NM', 'AZ', 'NV'],
            'west': ['CA', 'OR', 'WA', 'ID', 'MT', 'WY', 'CO', 'UT', 'AK', 'HI']
        }
        
        for region, states in regions.items():
            if state in states:
                return region
        return 'other'
    
    def generate_metadata_profile(self, interviewee_id: str) -> Dict[str, Any]:
        age = random.choice(self.demographics['ages'])
        gender = random.choice(self.demographics['genders'])
        name = random.choice(self.demographics['names'][gender])
        
        race_count = random.randint(1, 2)
        if race_count == 1:
            race = random.choices(
                self.demographics['races'], 
                weights=[40, 15, 20, 15, 5, 5], 
                k=1
            )
        else:
            available_races = self.demographics['races'].copy()
            race = []
            first_race = random.choices(
                available_races, 
                weights=[40, 15, 20, 15, 5, 5], 
                k=1
            )[0]
            race.append(first_race)
            available_races.remove(first_race)
            remaining_weights = [40, 15, 20, 15, 5, 5]
            race_index = self.demographics['races'].index(first_race)
            remaining_weights.pop(race_index)
            
            second_race = random.choices(available_races, weights=remaining_weights, k=1)[0]
            race.append(second_race)

        education = random.choice(self.demographics['education_levels'])
        employment = random.choice(self.demographics['employment_statuses'])
        income = random.choice(self.demographics['income_ranges'])
        city, state = random.choice(self.demographics['cities'])
        
        if employment in ['unemployed', 'retired', 'student']:
            occupation = employment
            industry = 'none'
        else:
            industry = random.choice(list(self.demographics['occupations'].keys()))
            occupation = random.choice(self.demographics['occupations'][industry])
        
        num_interviews = random.randint(1, 4)
        interviews = []
        
        base_date = datetime(2024, 1, 1)
        for i in range(num_interviews):
            interview_date = base_date + timedelta(days=random.randint(0, 365))
            interview_types = ['initial_screening', 'follow_up', 'product_feedback', 'user_experience']
            
            interview = {
                "interview_id": f"IV_{interviewee_id.split('_')[1]}_{interview_date.strftime('%Y%m%d')}",
                "date": interview_date.strftime('%Y-%m-%d'),
                "start_time": interview_date.replace(
                    hour=random.randint(9, 17), 
                    minute=random.choice([0, 15, 30, 45])
                ).strftime('%Y-%m-%dT%H:%M:%SZ'),
                "end_time": (interview_date.replace(
                    hour=random.randint(9, 17), 
                    minute=random.choice([0, 15, 30, 45])
                ) + timedelta(minutes=random.randint(30, 90))).strftime('%Y-%m-%dT%H:%M:%SZ'),
                "duration_minutes": random.randint(30, 90),
                "interview_type": random.choice(interview_types),
                "interviewer_id": random.choice(self.interviewers),
                "location": random.choice(['remote', 'in_person']),
                "recording_quality": random.choices(['excellent', 'good', 'fair'], weights=[50, 35, 15])[0],
                "transcript_confidence_score": random.uniform(0.75, 0.98),
                "completion_status": "completed",
                "transcript_file": f"transcript_IV_{interviewee_id.split('_')[1]}_{interview_date.strftime('%Y%m%d')}.json",
                "compensation_amount": random.choice([25, 50, 75, 100])
            }
            interviews.append(interview)
        
        total_compensation = sum(interview['compensation_amount'] for interview in interviews)
        total_duration = sum(interview['duration_minutes'] for interview in interviews)
        hourly_rate = round(total_compensation / (total_duration / 60), 2) if total_duration > 0 else 0
        
        metadata = {
            "interviewee_id": interviewee_id,
            "personal_info": {
                "name": name,
                "age": age,
                "age_range": self.generate_age_range(age),
                "gender": gender,
                "race_ethnicity": race,
                "education_level": education,
                "employment_status": employment,
                "occupation": occupation,
                "industry": industry,
                "income_range": income,
                "location": {
                    "city": city,
                    "state": state,
                    "country": "US",
                    "region": self.get_region(state)
                }
            },
            "interviews": interviews,
            "study_participation": {
                "recruitment_channel": random.choice(['email_campaign', 'social_media', 'referral', 'website']),
                "total_interviews_completed": len(interviews),
                "total_compensation": total_compensation,
                "hourly_compensation_rate": hourly_rate
            }
        }
        
        return metadata
    
    def generate_all_metadata(self) -> List[Dict[str, Any]]:
        print("Generating metadata for 100 people...")
        metadata_profiles = []
        
        for i in range(1, 101):
            interviewee_id = f"INT_{str(i).zfill(3)}"
            profile = self.generate_metadata_profile(interviewee_id)
            metadata_profiles.append(profile)
            
            with open(f'metadata/metadata_{interviewee_id}.json', 'w') as f:
                json.dump(profile, f, indent=2)
        
        print("Generated 100 metadata profiles")
        return metadata_profiles
    
    def generate_reviews_for_product(self, product: Dict[str, Any]) -> List[Dict[str, Any]]:
        print(f"Generating reviews for: {product['product_name']}")
        
        prompt = f"""For this product, generate 20 reviews with the most content/elaborations and of the GREATEST length:

{product['product_name']}: {product['description']}

{product['link']}

Requirements:
- I want to see an even distribution of reviews from 1 to 5 stars (4 reviews for each star rating)
- Each complete review should be 300 words MINIMUM
- Make reviews realistic with specific details, use cases, pros/cons
- Include varied demographics and use cases
- Make them feel authentic with personal experiences
- Include both positive and negative aspects even in good reviews
- Use natural language and varied writing styles

Format each review as:
Rating: [1-5] stars
Title: [Review title]
Review: [Full review text 300+ words]
Reviewer Profile: [Brief description of reviewer type/demographics]
---

Generate all 20 reviews now."""

        try:
            response = self.model.generate_content(prompt)
            reviews_text = response.text
            
            reviews = self.parse_reviews_response(reviews_text, product)
            
            with open(f"reviews/reviews_{product['id']:03d}.json", 'w') as f:
                json.dump({
                    'product': product,
                    'reviews': reviews
                }, f, indent=2)
            
            print(f"Generated {len(reviews)} reviews for {product['product_name']}")
            return reviews
            
        except Exception as e:
            print(f"Error generating reviews for {product['product_name']}: {e}")
            return []
    
    def parse_reviews_response(self, response_text: str, product: Dict[str, Any]) -> List[Dict[str, Any]]:
        reviews = []
        review_sections = response_text.split('---')
        
        for i, section in enumerate(review_sections):
            if not section.strip():
                continue
                
            try:
                lines = section.strip().split('\n')
                rating = None
                title = ""
                review_text = ""
                reviewer_profile = ""
                
                current_field = None
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('Rating:'):
                        rating_text = line.replace('Rating:', '').strip()
                        rating = int(''.join(filter(str.isdigit, rating_text)))
                        current_field = 'rating'
                    elif line.startswith('Title:'):
                        title = line.replace('Title:', '').strip()
                        current_field = 'title'
                    elif line.startswith('Review:'):
                        review_text = line.replace('Review:', '').strip()
                        current_field = 'review'
                    elif line.startswith('Reviewer Profile:'):
                        reviewer_profile = line.replace('Reviewer Profile:', '').strip()
                        current_field = 'profile'
                    elif current_field == 'review' and line:
                        review_text += " " + line
                    elif current_field == 'profile' and line:
                        reviewer_profile += " " + line
                
                if rating and review_text and len(review_text) > 100:
                    reviews.append({
                        'review_id': f"REV_{product['id']:03d}_{i+1:02d}",
                        'product_id': product['id'],
                        'rating': rating,
                        'title': title,
                        'review_text': review_text,
                        'reviewer_profile': reviewer_profile,
                        'word_count': len(review_text.split())
                    })
                    
            except Exception as e:
                print(f"Error parsing review section: {e}")
                continue
        
        return reviews
    
    def select_best_reviews(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        selected_reviews = []
        
        reviews_by_rating = {}
        for review in reviews:
            rating = review['rating']
            if rating not in reviews_by_rating:
                reviews_by_rating[rating] = []
            reviews_by_rating[rating].append(review)
        
        for rating in range(1, 6):
            if rating in reviews_by_rating:
                rating_reviews = sorted(
                    reviews_by_rating[rating], 
                    key=lambda x: x['word_count'], 
                    reverse=True
                )
                selected_reviews.extend(rating_reviews[:2])
        
        return selected_reviews
    
    def match_review_to_metadata(self, review: Dict[str, Any], metadata_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        product_id = review['product_id']
        product = next((p for p in self.products if p['id'] == product_id), None)
        
        if not product:
            return random.choice(metadata_profiles)
        
        tech_products = ['headphones', 'robot', 'projector', 'scanner', 'usb', 'gaming', 'smart', 'bluetooth']
        fitness_products = ['yoga', 'exercise', 'fitness', 'foam roller', 'resistance', 'dumbbells']
        home_products = ['kitchen', 'cooker', 'fryer', 'kettle', 'vacuum', 'blanket']
        
        product_name_lower = product['product_name'].lower()
        
        suitable_profiles = []
        
        for profile in metadata_profiles:
            personal_info = profile['personal_info']
            age = personal_info['age']
            income = personal_info['income_range']
            education = personal_info['education_level']
            
            if any(tech_word in product_name_lower for tech_word in tech_products):
                if age < 45 and income in ['$50k-$75k', '$75k-$100k', '$100k-$150k', '$150k+']:
                    suitable_profiles.append(profile)
            elif any(fit_word in product_name_lower for fit_word in fitness_products):
                if 25 <= age <= 55:
                    suitable_profiles.append(profile)
            elif any(home_word in product_name_lower for home_word in home_products):
                if age >= 25:
                    suitable_profiles.append(profile)
            else:
                suitable_profiles.append(profile)
        
        if not suitable_profiles:
            suitable_profiles = metadata_profiles
        
        return random.choice(suitable_profiles)
    
    def generate_interview_transcript(self, review: Dict[str, Any], metadata_profile: Dict[str, Any]) -> Dict[str, Any]:
        product = next((p for p in self.products if p['id'] == review['product_id']), None)
        personal_info = metadata_profile['personal_info']
        
        interviewer_name = random.choice(['Sarah', 'Mike', 'Jessica', 'David', 'Emily', 'Chris'])
        interviewee_name = personal_info['name']
        
        prompt = f"""TASK: Generate a customer interview based on the review below. 

INTERVIEW PARAMETERS:
- Length: ~1500-2500 tokens
- Interviewer: Corporate representative named {interviewer_name}
- Interviewee: Customer named {interviewee_name} 
- Focus: Product feedback, improvement suggestions, purchase motivations, user experience

CUSTOMER PROFILE:
- Name: {interviewee_name}
- Age: {personal_info['age']} ({personal_info['age_range']})
- Gender: {personal_info['gender']}
- Education: {personal_info['education_level']}
- Occupation: {personal_info['occupation']}
- Income: {personal_info['income_range']}
- Location: {personal_info['location']['city']}, {personal_info['location']['state']}

PRODUCT CONTEXT:
{product['product_name']}: {product['description']}

CUSTOMER REVIEW (Rating: {review['rating']}/5):
Title: {review['title']}
Review: {review['review_text']}

INTERVIEW GUIDELINES:
- Use natural, conversational dialogue with realistic interruptions and follow-ups
- Interviewer should probe deeper into specific pain points and positive aspects mentioned in the review
- Include between 15-25 back-and-forth exchanges
- Use realistic hesitations, "um's, and natural speech patterns
- Interviewer should be empathetic but professionally curious
- Avoid leading questions; use open-ended follow-ups
- Customer should feel comfortable sharing both positive and negative feedback
- End with a satisfaction rating request (1-10 scale)
- Make sure the customer's responses align with their demographic profile
- Do not publicate or repeat parts of the conversation
- Include natural interruptions and clarifications
- Vary response lengths (some short "yeah" responses, others longer explanations)
- Add realistic memory gaps ("I can't remember exactly, but...")
- Include contradictions or evolving opinions during the conversation
- IMPORTANT: Use the exact names {interviewer_name} and {interviewee_name} consistently throughout the dialogue

CORE QUESTION AREAS TO EXPLORE:
- What motivated the original purchase decision vs. alternatives
- Specific use cases and frequency of use
- Comparison to previous solutions/methods they used
- What exceeded expectations and what disappointed
- Household context and user scenarios
- Price/value perception
- Likelihood to recommend and reasons why/why not
- Specific suggestions for product improvements
- What would make them switch to a competitor

TRANSCRIPT RULES:
- NO UNICODE is to be in the transcript
- DO not generate * in any part of the transcript

INSTRUCTIONS: Create a realistic interview dialogue that captures the customer's experience authentically. The interviewer should follow up on specific details mentioned in the review and explore the broader context of the customer's needs and satisfaction. Make sure to use the specified names ({interviewer_name} and {interviewee_name}) consistently.

Format as natural dialogue with speaker names, no special formatting needed."""

        try:
            response = self.model.generate_content(prompt)
            transcript_text = response.text.encode().decode('unicode_escape')
            
            transcript_data = self.parse_transcript(transcript_text, review, metadata_profile, product, interviewer_name, interviewee_name)
            
            return transcript_data
            
        except Exception as e:
            print(f"Error generating transcript: {e}")
            return None
    
    def parse_transcript(self, transcript_text: str, review: Dict[str, Any], metadata_profile: Dict[str, Any], product: Dict[str, Any], interviewer_name: str, interviewee_name: str) -> Dict[str, Any]:
        lines = transcript_text.split('\n')
        turns = []
        turn_id = 1
        
        current_speaker = None
        current_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if any(keyword in line.lower() for keyword in ['interviewer', 'interviewee', 'date', 'corporate representative', 'customer']):
                continue
                
            if ':' in line:
                potential_speaker = line.split(':')[0].strip()
                
                potential_speaker = potential_speaker.replace('*', '').strip()
                
                if len(potential_speaker.split()) <= 2 and len(potential_speaker) < 20:
                    if current_speaker and current_text and len(current_text.strip()) > 5:
                        turns.append({
                            "turnId": turn_id,
                            "speaker": current_speaker,
                            "text": current_text.strip()
                        })
                        turn_id += 1
                    
                    current_speaker = potential_speaker
                    current_text = line.split(':', 1)[1].strip() if ':' in line else ""
                else:
                    if current_text:
                        current_text += " " + line
            else:
                if current_text:
                    current_text += " " + line
        
        if current_speaker and current_text and len(current_text.strip()) > 5:
            turns.append({
                "turnId": turn_id,
                "speaker": current_speaker,
                "text": current_text.strip()
            })
        
        interview_date = datetime.now() - timedelta(days=random.randint(1, 30))
        
        transcript_data = {
            "interviewId": f"product-interview-{review['review_id']}",
            "metadata": {
                "product": product['product_name'],
                "interviewDate": interview_date.strftime('%Y-%m-%d'),
                "interviewer": interviewer_name,
                "interviewee": f"{interviewee_name} ({metadata_profile['interviewee_id']})",
                "source": "synthetic_product_review"
            },
            "transcript": turns
        }
        
        return transcript_data
    
    def process_all_products(self, metadata_profiles: List[Dict[str, Any]]):
        print(f"Processing {len(self.products)} products...")
        
        for i, product in enumerate(self.products, 1):
            print(f"\n--- Processing Product {i}/{len(self.products)} ---")
            
            all_reviews = self.generate_reviews_for_product(product)
            
            if not all_reviews:
                print(f"Skipping {product['product_name']} - no reviews generated")
                continue
            
            selected_reviews = self.select_best_reviews(all_reviews)
            
            print(f"Selected {len(selected_reviews)} reviews for transcript generation")
            
            for j, review in enumerate(selected_reviews):
                matched_profile = self.match_review_to_metadata(review, metadata_profiles)
                
                transcript = self.generate_interview_transcript(review, matched_profile)
                
                if transcript:
                    transcript_filename = f"transcripts/transcript_{product['id']:03d}_{j+1:02d}.json"
                    with open(transcript_filename, 'w') as f:
                        json.dump(transcript, f, indent=2)
                    
                    print(f"Generated transcript {j+1}/{len(selected_reviews)}")
                else:
                    print(f"Failed to generate transcript {j+1}/{len(selected_reviews)}")
                
                time.sleep(2)
            
            time.sleep(5)
        
        print("\nAll products processed successfully!")
    
    def run_full_pipeline(self):
        print("Starting Synthetic Interview Data Generation Pipeline")
        print("=" * 60)
        
        metadata_profiles = self.generate_all_metadata()
        
        self.process_all_products(metadata_profiles)
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print(f"Generated files:")
        print(f"- 100 metadata profiles in /metadata/")
        print(f"- {len(self.products)} review sets in /reviews/")
        print(f"- ~{len(self.products) * 10} interview transcripts in /transcripts/")

if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.run_full_pipeline()