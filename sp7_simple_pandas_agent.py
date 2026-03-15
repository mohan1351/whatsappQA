"""
Simple Pandas Agent using OpenAI via LangChain
Executes analytical queries on WhatsApp DataFrame
"""

import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Dict
import os

# Load environment variables
load_dotenv()


class SimplePandasAgent:
    """
    Simple pandas agent for analytical queries
    Uses OpenAI GPT-4o-mini via LangChain
    """
    
    def __init__(self, df: pd.DataFrame, model: str = "gpt-4o-mini"):
        """
        Initialize pandas agent
        
        Args:
            df: WhatsApp DataFrame
            model: OpenAI model to use
        """
        self.df = df
        
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Build DataFrame schema description
        schema_description = self._build_schema_description(df)
        
        # Create pandas agent with schema context
        self.agent = create_pandas_dataframe_agent(
            self.llm,
            df,
            verbose=True,  # Show intermediate steps
            allow_dangerous_code=True,  # Required for code execution
            agent_type="openai-tools",
            return_intermediate_steps=True,  # Capture generated code
            prefix=schema_description  # Add schema context
        )
        
        print(f"✅ Pandas Agent initialized with {len(df)} messages")
    
    def _build_schema_description(self, df: pd.DataFrame) -> str:
        """
        Build detailed schema description for the agent
        
        Args:
            df: WhatsApp DataFrame
        
        Returns:
            Schema description string
        """
        description = f"""
You are working with a WhatsApp chat DataFrame with {len(df)} messages.

⚠️ CRITICAL RULES - READ FIRST:
1. ONLY answer based on data in this DataFrame - DO NOT use external knowledge
2. If data is not found, say "No data found in messages" - DO NOT make up answers
3. DO NOT use your training data about dates, events, or general facts
4. ONLY report what exists in the actual message_english content
5. ⚠️ HANDLE NEGATION QUERIES SPECIALLY - See NEGATION section below

CRITICAL SCHEMA INFORMATION:

KEY COLUMNS:
- sender: Person who SENT the message (use for "from X", "X sent", "who sent")
- birthday_person: Person whose BIRTHDAY DATE this message was sent on
  ⚠️ IMPORTANT: This marks the DATE (whose birthday it was)
  ⚠️ To find actual birthday WISHES, search message_english for birthday keywords
  
- message_english: English message text 
  🔍 CRITICAL: ALWAYS use THIS column for text searches (str.contains, keyword searches)
  ⚠️ DO NOT use 'message' column - it may have different encoding/format
- year: Year of message (integer: 2023, 2024, 2025, etc.)
- month: Month of message (integer: 1-12)
- dt: Full timestamp/date of message

NEGATION QUERIES - SPECIAL HANDLING ⚠️:

⭐ NEGATION DETECTION - Look for these words:
  - "NOT", "DIDN'T", "NEVER", "DIDN'T", "DON'T"
  - "EXCLUDING", "EXCEPT", "WITHOUT"
  - "who did not", "who didn't", "nobody", "no one"
  
NEGATION LOGIC - How to handle:
1. First, find who DID the action (positive query)
2. Get list of ALL senders
3. Use SET SUBTRACTION: all_senders - did_action = didn't_do_action
4. Return the NON-MATCHING senders list, clearly stating "These people did NOT: [list]"

EXAMPLE NEGATION QUERIES:

1. "who did NOT wish Krati birthday?"
   ✅ CORRECT:
   all_senders = df['sender'].unique()
   wished = df[(df['birthday_person'] == 'Krati') & (df['message_english'].str.contains('birth|wish|happy', case=False, na=False))]['sender'].unique()
   did_not_wish = list(set(all_senders) - set(wished))
   # Return: who did NOT wish
   
   ❌ WRONG: df[(df['birthday_person'] == 'Krati')]['sender'].unique() 
   # Returns who WISHED (opposite of what's asked)

2. "who didn't send messages in January?"
   ✅ CORRECT:
   all_senders = df['sender'].unique()
   sent_in_jan = df[df['month'] == 1]['sender'].unique()
   didnt_send = list(set(all_senders) - set(sent_in_jan))
   # Return: who DIDN'T send in January

3. "who never mentioned 'food'?"
   ✅ CORRECT:
   all_senders = df['sender'].unique()
   mentioned_food = df[df['message_english'].str.contains('food', case=False, na=False)]['sender'].unique()
   never_mentioned = list(set(all_senders) - set(mentioned_food))
   # Return: who NEVER mentioned food

4. "message count excluding Amit"
   ✅ CORRECT:
   df[df['sender'] != 'Amit']  # Exclude Amit, show rest
   # Count the rest

HOW TO RETURN NEGATION ANSWER:
- Use Python code like above
- Return the COMPLEMENT/INVERSE set
- Make answer clear: "These people did NOT: [list]"

EXAMPLE QUERIES AND CORRECT CODE:

1. "who sent birthday wishes to Krati?"
   ✅ CORRECT: df[(df['birthday_person'] == 'Krati') & (df['message_english'].str.contains('birth|wish|happy', case=False, na=False))]['sender'].unique()
   ❌ WRONG: df[df['birthday_person'] == 'Krati']['sender'].unique()  # includes ALL messages on that date

2. "who mentioned 'Golgappa' in 2024?"
   ✅ CORRECT: df[df['message_english'].str.contains('Golgappa', na=False) & (df['year'] == 2024)]
   ❌ WRONG: df[df['message'].str.contains('Golgappa')]  # NEVER use 'message' column

3. "how many birthday messages did Priya receive?"
   ✅ CORRECT: len(df[(df['birthday_person'] == 'Priya') & (df['message_english'].str.contains('birth|wish|happy', case=False, na=False))])
   ❌ WRONG: len(df[df['birthday_person'] == 'Priya'])  # includes ALL messages on birthday

4. "messages from Amit in 2024"
   ✅ CORRECT: df[(df['sender'] == 'Amit') & (df['year'] == 2024)]

5. "how many messages were sent on Krati's birthday?" (all messages, not just wishes)
   ✅ CORRECT: len(df[df['birthday_person'] == 'Krati'])  # ALL messages on that date

RULES:
- ⚠️ CRITICAL: ONLY answer based on DataFrame content - NO external knowledge or assumptions
- ⚠️ NEGATION QUERIES: Always use set subtraction (all - matching = non-matching)
- ⚠️ If no data found, respond: "No relevant data found in the messages"
- 🔍 TEXT SEARCHES: ALWAYS use message_english column with .str.contains() (set na=False)
- For birthday WISHES: Use birthday_person column + text search in message_english for keywords (birth, wish, happy)
- birthday_person alone = all messages on that date (not just wishes)
- Use sender column for who sent messages
- For negation ("who did NOT"), use set subtraction
- For counting, use len() or value_counts()
- For unique values, use .unique()
- NEVER use 'message' column for text search - ONLY use 'message_english'

Available columns: {list(df.columns)}
"""
        return description
    
    def query(self, question: str) -> Dict[str, str]:
        """
        Execute analytical query
        
        Args:
            question: Natural language query
        
        Returns:
            Dict with 'answer' and 'code' keys
        """
        try:
            # Run agent
            result = self.agent.invoke({"input": question})
            
            # Extract output
            if isinstance(result, dict):
                answer = result.get('output', str(result))
                intermediate_steps = result.get('intermediate_steps', [])
            else:
                answer = str(result)
                intermediate_steps = []
            
            # Extract pandas code from intermediate steps
            generated_code = []
            for step in intermediate_steps:
                if isinstance(step, tuple) and len(step) >= 2:
                    # step is (AgentAction, observation)
                    action = step[0]
                    if hasattr(action, 'tool_input'):
                        code = action.tool_input
                        if isinstance(code, dict):
                            code = code.get('query', str(code))
                        if code and 'df' in str(code):
                            generated_code.append(str(code))
            
            return {
                'answer': answer,
                'code': '\n'.join(generated_code) if generated_code else 'No code generated'
            }
            
        except Exception as e:
            return {
                'answer': f"❌ Error executing query: {e}",
                'code': 'Error - no code generated'
            }


def demo():
    """
    Demo pandas agent with WhatsApp data
    """
    print("="*80)
    print("SIMPLE PANDAS AGENT DEMO")
    print("="*80)
    
    # Load data
    print("\n[1/2] Loading WhatsApp data...")
    df = pd.read_csv("data/whatsapp/parivar/parivar_features.csv", encoding="utf-8-sig")
    df_filtered = df[df['message_english'].notna() & (df['message_english'].str.len() >= 3)]
    print(f"✅ Loaded {len(df_filtered)} messages")
    
    # Initialize agent
    print("\n[2/2] Initializing pandas agent...")
    agent = SimplePandasAgent(df_filtered)
    
    # Test queries
    test_queries = [
        "How many messages are in the dataset?",
        "Who sent the most messages?",
        "How many unique senders are there?",
        "Count messages by year",
        "List all unique senders"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        result = agent.query(query)
        print(f"\n🐍 Generated Code:\n{result['code']}")
        print(f"\n📊 Answer:\n{result['answer']}")


if __name__ == "__main__":
    try:
        demo()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
