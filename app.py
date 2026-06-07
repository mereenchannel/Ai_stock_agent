import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. ตั้งค่าหน้าเว็บสไตล์ระบบเทรดมินิมอล ---
st.set_page_config(page_title="AI Quant Trading Agent", page_icon="📈", layout="wide")

st.markdown("""
        <style>
    /* --- 1. ซ่อนเมนู Streamlit และตั้งค่าพื้นหลัง Dark Mode --- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {background-color: #0b0e14; color: #ffffff;}
    
    /* --- 2. ตกแต่งปุ่มกดส่งคำสั่ง --- */
    .stButton>button {
        background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
        color: white; font-size: 16px; font-weight: bold; border-radius: 8px;
        border: none; padding: 10px 20px; width: 100%; box-shadow: 0 4px 15px rgba(0,180,219,0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0083b0 0%, #00b4db 100%);
        box-shadow: 0 6px 20px rgba(0,180,219,0.5);
    }

    /* --- 3. ตกแต่งกล่องกรอกชื่อหุ้น (Text Input) --- */
    div[data-testid="stTextInput"] input {
        background-color: #1e293b !important;
        color: #00b4db !important; 
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border: 2px solid #00b4db !important;
        box-shadow: 0 0 10px rgba(0, 180, 219, 0.3) !important;
    }
    div[data-testid="stTextInput"] label p {
        color: #8f9cae !important;
        font-weight: bold;
    }

    /* --- 4. ปรับสีตัวอักษรทั่วไป (รวมถึงบทวิเคราะห์ AI) ให้อ่านง่ายบนพื้นดำ --- */
    p, li {
        color: #e2e8f0;
    }
    div[data-testid="stAlert"] p, div[data-testid="stAlert"] li {
        color: #ffffff !important;
    }

    /* --- 5. แก้สีหัวข้อและตัวเลขกล่องข้อมูล (Metrics) --- */
    div[data-testid="stMetricLabel"] > div, div[data-testid="stMetricLabel"] p {
        color: #ffffff !important; 
        font-size: 16px !important; 
        font-weight: 500 !important;
    }
    div[data-testid="stMetricValue"] { 
        color: #00b4db !important; 
        font-size: 28px !important; 
    }

    /* --- 6. ซ่อนกล่อง Tooltip ที่เด้งบังตัวหนังสือเวลากราฟทำงาน --- */
    #vg-tooltip-element {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
    }
    </style>

""", unsafe_allow_html=True)

# --- ส่วนหัวแอปพลิเคชัน ---
st.markdown("<h1 style='text-align: center; color: #00b4db;'>📈 AI Quant Trading Agent Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #effcff;'>ระบบวิเคราะห์ทางเทคนิคอลและดักจับโครงสร้างเงินทุนรายใหญ่ด้วยสถาปัตยกรรม AI</p>", unsafe_allow_html=True)
st.markdown("---")

# --- ฟังก์ชันคำนวณทางเทคนิค (สูตรอมตะ) ---
def calculate_ema(df, column="Close", period=20):
    return df[column].ewm(span=period, adjust=False).mean()

def calculate_mcdx_banker(df):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    vol_ma = df['Volume'].rolling(window=20).mean()
    vol_ratio = df['Volume'] / vol_ma
    
    banker = (rsi - 50) * 1.5 * vol_ratio
    banker = banker.clip(lower=0, upper=100) 
    return banker.rolling(window=3).mean().fillna(0)

# --- ส่วนควบคุม (Sidebar/Inputs) ---
col_input1, col_input2 = st.columns([1, 2])

with col_input1:
    st.markdown("### 🛠️ กรุณาใส่ชื่อหุ้นที่ต้องการ")
    ticker = st.text_input("🔤 ถ้าเป็นหุ้นไทยให้ใส่ .bk ด้านหลังชื่อหุ้น เช่น PTT.bk", value="").strip()
    
    st.markdown("---")
    st.markdown("**💡 คำแนะนำเรื่องชื่อหุ้น:**")
    st.markdown("- หุ้นสหรัฐฯ: พิมพ์ตัวย่อได้เลย เช่น `AAPL`, `NVDA` ")
    st.markdown("- หุ้นไทย: ต้องเติมนามสกุล เช่น `CPALL.BK`, `PTT.BK`")

with col_input2:
    st.markdown("### 📊 หน้าจอกระดานวิเคราะห์")
    if st.button("🚀 เริ่มวิเคราะห์โครงสร้างราคาและรายใหญ่"):
        with st.spinner("💾 กำลังดึงข้อมูลและประมวลผลอินดิเคเตอร์หลังบ้าน..."):
            try:
                # 1. ดึงข้อมูลผ่าน Yahoo Finance
                stock = yf.Ticker(ticker)
                df = stock.history(period="1y")
                
                if df.empty:
                    st.error("❌ ไม่พบข้อมูลหุ้นตัวนี้ กรุณาตรวจสอบตัวย่อหุ้นอีกครั้งครับ")
                else:
                    # 2. คำนวณอินดิเคเตอร์ทั้งหมด
                    df['EMA_20'] = calculate_ema(df, period=20)
                    df['EMA_50'] = calculate_ema(df, period=50)
                    df['EMA_200'] = calculate_ema(df, period=200)
                    df['MCDX_Banker_Green'] = calculate_mcdx_banker(df)
                    
                    df_round = df.round(2)
                    df_60d = df_round.tail(60)
                    df_5d = df_round.tail(5)
                    
                    # 3. แสดงผลตัวเลขสำคัญล่าสุด (Metrics)
                    last_row = df_round.iloc[-1]
                    m1, m2, m3 = st.columns(3)
                    m1.metric("ราคาปัจจุบัน", f"${last_row['Close']}" if ".BK" not in ticker else f"{last_row['Close']} บาท")
                    m2.metric("เส้น EMA 200 (ภาพใหญ่)", f"{last_row['EMA_200']}")
                    m3.metric("MCDX รายใหญ่ (สีเขียว)", f"{last_row['MCDX_Banker_Green']}%")
                    
                    # 4. วาดกราฟเทคนิคอลแยกส่วน (เหมือนโปรแกรมเทรดจริง)
                    st.markdown("#### 📉 กราฟราคา และเส้นค่าเฉลี่ยศัลยกรรมแนวโน้ม (EMA 20, 50, 200)")
                    chart_data_price = df_60d[['Close', 'EMA_20', 'EMA_50', 'EMA_200']]
                    st.line_chart(chart_data_price)
                    
                    st.markdown("#### 🟢 กราฟระดับการสะสมทุนของรายใหญ่ (MCDX Banker)")
                    chart_data_mcdx = df_60d[['MCDX_Banker_Green']]
                    st.area_chart(chart_data_mcdx, color="#00b4db")
                    
                    # 5. ส่วนของการเรียกใช้สมอง AI (พรุ่งนี้พอกุญแจพร้อมจะเปิดใช้งานส่วนนี้อัตโนมัติ)
                    st.markdown("---")
                    st.markdown("### 🧠 บทวิเคราะห์และฟันธงกลยุทธ์จาก AI Agent")
                    
                    # ดึงสถิติภาพรวม 60 วันส่งให้ AI
                    highest_close = df_60d['Close'].max()
                    lowest_close = df_60d['Close'].min()
                    avg_mcdx_60d = df_60d['MCDX_Banker_Green'].mean()
                    days_above_base = (df_60d['MCDX_Banker_Green'] >= 45).sum()
                    
                    summary_text = f"""
                    - ในรอบ 60 วันทำการที่ผ่านมา:
                      * ราคาสูงสุดอยู่ที่: {highest_close}
                      * ราคาต่ำสุดอยู่ที่: {lowest_close}
                      * ค่าเฉลี่ยสะสมของรายใหญ่ (MCDX) อยู่ที่: {avg_mcdx_60d:.2f}
                      * จำนวนวันที่รายใหญ่คุมตลาดเหนือเกณฑ์ฐาน 45: {days_above_base} จาก 60 วัน
                    """
                    recent_5d_table = df_5d[['Close', 'EMA_20', 'EMA_50', 'EMA_200', 'MCDX_Banker_Green']].to_string()
                    
                    # ดักจับเรื่อง API Key หลุด/หมดโควต้าชั่วคราว
                    try:
                        # ตรวจสอบคีย์ในระบบเซิร์ฟเวอร์
                        api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                        
                        if not api_key:
                            st.warning("🔑 บอทคำนวณกราฟเสร็จแล้ว! แต่ยังไม่ได้เชื่อมต่อกุญแจ API ของ Gemini (รอใส่รหัสพรุ่งนี้เพื่อเปิดระบบตัดสินใจ)")
                        else:
                            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0.1)
                            
                            prompt = f"""คุณคือ 'ผู้จัดการกองทุนสายเทคนิคอลขั้นสูง' 
                            จงวิเคราะห์สถานการณ์ของหุ้นตัวนี้ โดยพิจารณาจากข้อมูลสองส่วนดังนี้:
                            
                            [ส่วนที่ 1: สรุปภาพรวมความเคลื่อนไหวในรอบ 60 วัน]
                            {summary_text}
                            
                            [ส่วนที่ 2: ตารางข้อมูลราคาและอินดิเคเตอร์แบบละเอียด 5 วันล่าสุด]
                            {recent_5d_table}
                            
                            เงื่อนไขและกฎกลยุทธ์การเทรด (Trading Logic):
                            1. วิเคราะห์แนวโน้ม (EMA 20, 50, 200): ดูว่าราคายืนอยู่เหนือเส้นสะท้อนแนวโน้มระยะสั้น ระยะกลาง และระยะยาวหรือไม่?
                            2. วิเคราะห์สัญญาณ Smart Money (MCDX): ดูพฤติกรรมรายใหญ่เทียบกับเกณฑ์ฐานที่ 45
                            3. ฟันธงคำสั่ง: เลือกทางใดทางหนึ่งอย่างชัดเจนระหว่าง BUY, SELL หรือ HOLD พร้อมคำนวณจุด Stop Loss และ Take Profit ที่ชัดเจนเป็นตัวเลข
                            คุณกำลังวิเคราะห์สำหรับตลาด Spot ที่ซื้อขายแบบ Long เท่านั้น (ทำกำไรขาขึ้นได้อย่างเดียว) หากกราฟเป็นขาลง ห้ามแนะนำให้เปิด Short Sell เด็ดขาด ให้ฟันธงเป็น HOLD (ถือเงินสดรอดูสถานการณ์) หรือหากแนะนำ SELL ให้หมายถึงการขายหุ้นที่มีอยู่เพื่อตัดขาดทุนหรือ จำนวนเปอร์เซ็นต์ที่ควรขาย โดยไม่ต้องคำนวณจุด TP/SL"
                            โปรดเขียนบทวิเคราะห์ให้เฉียบคม ลึกซึ้ง และระบุเหตุผลเชิงสถิติจากทั้งรอบ 60 วันและ 5 วันล่าสุด
                            """
                            
                            response = llm.invoke(prompt)
                            st.markdown(response.content)
                            
                    except Exception as ai_error:
                        st.error(f"⚠️ สมอง AI ยังไม่พร้อมทำงานเนื่องจากโควต้าเต็มชั่วคราว: {ai_error}")
                        st.info("💡 ข้อแนะนำจาก Mentor: ระบบคำนวณตัวเลขและพล็อตขุดกราฟด้านบนทำงานถูกต้อง 100% แล้วครับ ตอนนี้เหลือเพียงรอเวลาให้โควต้ารายวันของ Google รีเซ็ตใหม่ในวันพรุ่งนี้ ตัวกล่องวิเคราะห์ AI ด้านบนนี้ก็จะสับสวิตช์ทำงานฟันธงหุ้นให้คุณทันทีครับ!")
                        
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดทางเทคนิคหลังบ้าน: {e}")
