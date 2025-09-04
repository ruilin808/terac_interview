import json
import time
import threading
import queue
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from rag import QdrantRAG

class InterviewerStatus(Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    BREAK = "break"

class QueryPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class CustomerQuery:
    query_id: str
    customer_id: str
    query_text: str
    priority: QueryPriority
    timestamp: datetime
    expected_duration: int
    category: str
    metadata: Dict[str, Any]

@dataclass
class Interviewer:
    interviewer_id: str
    name: str
    specialties: List[str]
    status: InterviewerStatus
    current_load: int
    max_capacity: int
    hourly_rate: float
    availability_schedule: Dict[str, List[Tuple[str, str]]]
    performance_metrics: Dict[str, float]
    last_activity: datetime

@dataclass
class InterviewAssignment:
    assignment_id: str
    query: CustomerQuery
    interviewer: Interviewer
    target_interviewees: List[str]
    estimated_start_time: datetime
    estimated_completion_time: datetime
    status: str
    priority_score: float

class InterviewRoutingSystem:
    def __init__(self, rag_system: QdrantRAG):
        self.rag_system = rag_system
        self.interviewers = {}
        self.query_queue = queue.PriorityQueue()
        self.active_assignments = {}
        self.completed_assignments = {}
        
        self.total_queries_processed = 0
        self.average_wait_time = 0
        self.system_efficiency = 0
        
        self._initialize_interviewers()
        
        self.running = True
        self.processor_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processor_thread.start()
        
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
    
    def _initialize_interviewers(self):
        interviewer_data = [
            {
                "interviewer_id": "AI_TECH_001",
                "name": "TechBot Alpha",
                "specialties": ["technology", "gaming", "electronics", "headphones", "smart devices"],
                "hourly_rate": 45.0,
                "max_capacity": 8
            },
            {
                "interviewer_id": "AI_HEALTH_001", 
                "name": "HealthBot Beta",
                "specialties": ["healthcare", "fitness", "wellness", "toothbrush", "medical"],
                "hourly_rate": 50.0,
                "max_capacity": 6
            },
            {
                "interviewer_id": "AI_HOME_001",
                "name": "HomeBot Gamma",
                "specialties": ["home", "kitchen", "appliances", "airfryer", "cooker"],
                "hourly_rate": 40.0,
                "max_capacity": 10
            },
            {
                "interviewer_id": "AI_BIZ_001",
                "name": "BizBot Delta",
                "specialties": ["business", "productivity", "office", "scanner", "projector"],
                "hourly_rate": 55.0,
                "max_capacity": 5
            },
            {
                "interviewer_id": "AI_LIFESTYLE_001",
                "name": "LifestyleBot Epsilon",
                "specialties": ["retail", "fashion", "lifestyle", "watches", "accessories"],
                "hourly_rate": 35.0,
                "max_capacity": 8
            },
            {
                "interviewer_id": "AI_GENERAL_001",
                "name": "GeneralBot Zeta",
                "specialties": ["general", "consumer", "products", "battery", "reviews"],
                "hourly_rate": 30.0,
                "max_capacity": 12
            }
        ]
        
        for data in interviewer_data:
            interviewer = Interviewer(
                interviewer_id=data["interviewer_id"],
                name=data["name"],
                specialties=data["specialties"],
                status=InterviewerStatus.AVAILABLE,
                current_load=0,
                max_capacity=data["max_capacity"],
                hourly_rate=data["hourly_rate"],
                availability_schedule={
                    "monday": [("00:00", "23:59")],
                    "tuesday": [("00:00", "23:59")],
                    "wednesday": [("00:00", "23:59")],
                    "thursday": [("00:00", "23:59")],
                    "friday": [("00:00", "23:59")],
                    "saturday": [("00:00", "23:59")],
                    "sunday": [("00:00", "23:59")],
                },
                performance_metrics={
                    "avg_interview_duration": random.uniform(30, 50),
                    "customer_satisfaction": random.uniform(4.0, 4.9),
                    "completion_rate": random.uniform(0.95, 0.99)
                },
                last_activity=datetime.now()
            )
            self.interviewers[data["interviewer_id"]] = interviewer
    
    def submit_query(self, customer_id: str, query_text: str, 
                    priority: QueryPriority = QueryPriority.NORMAL,
                    expected_duration: int = 60,
                    category: str = "general") -> str:
        
        query = CustomerQuery(
            query_id=f"Q_{uuid.uuid4().hex[:8]}",
            customer_id=customer_id,
            query_text=query_text,
            priority=priority,
            timestamp=datetime.now(),
            expected_duration=expected_duration,
            category=category,
            metadata={}
        )
        
        self.query_queue.put((priority.value, query.timestamp, query))
        
        print(f"Query submitted: {query.query_id} - '{query_text[:50]}...'")
        return query.query_id
    
    def _process_query(self, query: CustomerQuery) -> InterviewAssignment:
        target_interviewees, detailed_results = self.rag_system.query(query.query_text)
        
        best_interviewer = self._find_best_interviewer(query, target_interviewees)
        
        if not best_interviewer:
            query.priority = QueryPriority.HIGH
            self.query_queue.put((query.priority.value, query.timestamp, query))
            return None
        
        priority_score = self._calculate_priority_score(query, best_interviewer)
        estimated_start = datetime.now() + timedelta(minutes=random.randint(5, 30))
        estimated_completion = estimated_start + timedelta(minutes=query.expected_duration)
        
        assignment = InterviewAssignment(
            assignment_id=f"A_{uuid.uuid4().hex[:8]}",
            query=query,
            interviewer=best_interviewer,
            target_interviewees=target_interviewees,
            estimated_start_time=estimated_start,
            estimated_completion_time=estimated_completion,
            status="assigned",
            priority_score=priority_score
        )
        
        best_interviewer.current_load += 1
        if best_interviewer.current_load >= best_interviewer.max_capacity:
            best_interviewer.status = InterviewerStatus.BUSY
        
        return assignment
    
    def _find_best_interviewer(self, query: CustomerQuery, target_interviewees: List[str]) -> Optional[Interviewer]:
        available_interviewers = [
            interviewer for interviewer in self.interviewers.values()
            if interviewer.status == InterviewerStatus.AVAILABLE and 
               interviewer.current_load < interviewer.max_capacity
        ]
        
        if not available_interviewers:
            return None
        
        scored_interviewers = []
        
        for interviewer in available_interviewers:
            score = 0
            
            query_words = query.query_text.lower().split()
            for specialty in interviewer.specialties:
                if any(specialty_word in query.query_text.lower() 
                      for specialty_word in specialty.split()):
                    score += 10
            
            load_factor = (interviewer.max_capacity - interviewer.current_load) / interviewer.max_capacity
            score += load_factor * 5
            
            score += interviewer.performance_metrics["customer_satisfaction"]
            score += interviewer.performance_metrics["completion_rate"] * 3
            
            if query.priority in [QueryPriority.HIGH, QueryPriority.URGENT]:
                if interviewer.performance_metrics["customer_satisfaction"] > 4.5:
                    score += 5
            
            cost_factor = (100 - interviewer.hourly_rate) / 100 * 2
            score += cost_factor
            
            scored_interviewers.append((score, interviewer))
        
        scored_interviewers.sort(reverse=True)
        return scored_interviewers[0][1]
    
    def _calculate_priority_score(self, query: CustomerQuery, interviewer: Interviewer) -> float:
        base_score = query.priority.value * 10
        
        wait_time = (datetime.now() - query.timestamp).total_seconds() / 60
        urgency_bonus = min(wait_time * 0.1, 5)
        
        quality_score = interviewer.performance_metrics["customer_satisfaction"] * 2
        
        return base_score + urgency_bonus + quality_score
    
    def _process_queue(self):
        while self.running:
            try:
                priority, timestamp, query = self.query_queue.get(timeout=1)
                
                print(f"Processing query: {query.query_id}")
                
                assignment = self._process_query(query)
                
                if assignment:
                    self.active_assignments[assignment.assignment_id] = assignment
                    self.total_queries_processed += 1
                    
                    print(f"Assignment created: {assignment.assignment_id}")
                    print(f"  Interviewer: {assignment.interviewer.name}")
                    print(f"  Target Interviewees: {', '.join(assignment.target_interviewees)}")
                    print(f"  Estimated Start: {assignment.estimated_start_time.strftime('%H:%M')}")
                    print()
                    
                    self._simulate_interview_progress(assignment)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing query: {e}")
    
    def _simulate_interview_progress(self, assignment: InterviewAssignment):
        def complete_interview():
            delay = random.randint(30, 90)
            time.sleep(delay)
            
            if assignment.assignment_id in self.active_assignments:
                del self.active_assignments[assignment.assignment_id]
                assignment.status = "completed"
                self.completed_assignments[assignment.assignment_id] = assignment
                
                assignment.interviewer.current_load -= 1
                if assignment.interviewer.current_load < assignment.interviewer.max_capacity:
                    assignment.interviewer.status = InterviewerStatus.AVAILABLE
                
                print(f"Interview completed: {assignment.assignment_id}")
        
        threading.Thread(target=complete_interview, daemon=True).start()
    
    def _monitor_system(self):
        while self.running:
            time.sleep(30)
            
            self._update_metrics()
            self._simulate_status_changes()
    
    def _update_metrics(self):
        total_capacity = sum(i.max_capacity for i in self.interviewers.values())
        current_load = sum(i.current_load for i in self.interviewers.values())
        
        self.system_efficiency = (current_load / total_capacity) * 100 if total_capacity > 0 else 0
        
        if self.active_assignments:
            wait_times = [
                (datetime.now() - assignment.query.timestamp).total_seconds() / 60
                for assignment in self.active_assignments.values()
            ]
            self.average_wait_time = sum(wait_times) / len(wait_times)
    
    def _simulate_status_changes(self):
        for interviewer in self.interviewers.values():
            if random.random() < 0.1:
                if interviewer.status == InterviewerStatus.AVAILABLE:
                    if random.random() < 0.3:
                        interviewer.status = InterviewerStatus.BREAK
                elif interviewer.status == InterviewerStatus.BREAK:
                    if random.random() < 0.7:
                        interviewer.status = InterviewerStatus.AVAILABLE
                elif interviewer.status == InterviewerStatus.BUSY:
                    if random.random() < 0.2 and interviewer.current_load > 0:
                        interviewer.current_load -= 1
                        if interviewer.current_load < interviewer.max_capacity:
                            interviewer.status = InterviewerStatus.AVAILABLE
    
    def get_system_status(self) -> Dict[str, Any]:
        available_interviewers = sum(1 for i in self.interviewers.values() 
                                   if i.status == InterviewerStatus.AVAILABLE)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_interviewers": len(self.interviewers),
            "available_interviewers": available_interviewers,
            "active_assignments": len(self.active_assignments),
            "completed_assignments": len(self.completed_assignments),
            "total_queries_processed": self.total_queries_processed,
            "queue_size": self.query_queue.qsize(),
            "system_efficiency": round(self.system_efficiency, 2),
            "average_wait_time": round(self.average_wait_time, 2),
            "interviewer_details": {
                interviewer_id: {
                    "name": interviewer.name,
                    "status": interviewer.status.value,
                    "current_load": interviewer.current_load,
                    "max_capacity": interviewer.max_capacity,
                    "specialties": interviewer.specialties
                }
                for interviewer_id, interviewer in self.interviewers.items()
            }
        }
    
    def get_assignment_details(self, assignment_id: str) -> Optional[Dict[str, Any]]:
        assignment = (self.active_assignments.get(assignment_id) or 
                     self.completed_assignments.get(assignment_id))
        
        if not assignment:
            return None
        
        return {
            "assignment_id": assignment.assignment_id,
            "query": {
                "query_id": assignment.query.query_id,
                "customer_id": assignment.query.customer_id,
                "query_text": assignment.query.query_text,
                "priority": assignment.query.priority.name,
                "timestamp": assignment.query.timestamp.isoformat()
            },
            "interviewer": {
                "interviewer_id": assignment.interviewer.interviewer_id,
                "name": assignment.interviewer.name,
                "specialties": assignment.interviewer.specialties
            },
            "target_interviewees": assignment.target_interviewees,
            "estimated_start_time": assignment.estimated_start_time.isoformat(),
            "estimated_completion_time": assignment.estimated_completion_time.isoformat(),
            "status": assignment.status,
            "priority_score": round(assignment.priority_score, 2)
        }
    
    def shutdown(self):
        print("Shutting down routing system...")
        self.running = False
        if self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2)
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        print("Routing system shutdown complete.")


def run_demo():
    print("Starting Dynamic Interview Routing System Demo")
    print("=" * 60)
    
    rag_system = QdrantRAG()
    routing_system = InterviewRoutingSystem(rag_system)
    
    test_queries = [
        ("CUST_001", "What are some favorites in the headphones category and what makes them successful", QueryPriority.NORMAL),
        ("CUST_002", "What do users think of my airfryer lineup of the brand COSORI", QueryPriority.HIGH),
        ("CUST_003", "What features do popular non-analog watches on the market have", QueryPriority.NORMAL),
        ("CUST_004", "How does battery life play into consumer appeal", QueryPriority.URGENT),
        ("CUST_005", "Why are electric toothbrushes popular", QueryPriority.LOW),
        ("CUST_006", "What makes a good fitness tracker", QueryPriority.NORMAL),
        ("CUST_007", "Kitchen appliance preferences for small apartments", QueryPriority.HIGH),
    ]
    
    submitted_queries = []
    
    print("\nSubmitting customer queries...")
    for customer_id, query_text, priority in test_queries:
        query_id = routing_system.submit_query(
            customer_id=customer_id,
            query_text=query_text,
            priority=priority,
            expected_duration=random.randint(45, 75)
        )
        submitted_queries.append(query_id)
        time.sleep(1)
    
    print("\nMonitoring system performance...")
    for i in range(6):
        time.sleep(30)
        status = routing_system.get_system_status()
        
        print(f"\nSystem Status (Check {i+1}/6):")
        print(f"  Active Assignments: {status['active_assignments']}")
        print(f"  Completed: {status['completed_assignments']}")
        print(f"  Queue Size: {status['queue_size']}")
        print(f"  Available Interviewers: {status['available_interviewers']}/{status['total_interviewers']}")
        print(f"  System Efficiency: {status['system_efficiency']}%")
        print(f"  Average Wait Time: {status['average_wait_time']} min")
    
    print("\nFinal System Report:")
    final_status = routing_system.get_system_status()
    print(json.dumps(final_status, indent=2))
    
    print("\nAssignment Details:")
    for assignment_id in list(routing_system.completed_assignments.keys())[:3]:
        details = routing_system.get_assignment_details(assignment_id)
        if details:
            print(f"\nAssignment {assignment_id}:")
            print(f"  Query: '{details['query']['query_text'][:60]}...'")
            print(f"  Interviewer: {details['interviewer']['name']}")
            print(f"  Target Interviewees: {', '.join(details['target_interviewees'])}")
    
    routing_system.shutdown()
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()