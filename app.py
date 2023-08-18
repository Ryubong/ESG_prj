import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, redirect, url_for
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import io
import base64
import pickle
import os

app = Flask(__name__, static_url_path='/static')

def save_model(model, model_name):
    with open(model_name, 'wb') as f:
        pickle.dump(model, f)

def load_model(model_name):
    with open(model_name, 'rb') as f:
        return pickle.load(f)

#환경 예측모델
def envi_process_data_and_predict(input_data, datafile, features, target):
    
     # PKL 파일 확인
    if os.path.exists("random_search_model.pkl") and os.path.exists("label_encoder.pkl"):
        random_search = load_model("random_search_model.pkl")
        le = load_model("label_encoder.pkl")
    else:
        random_search.fit(X_train, y_train)
        
        # pkl 파일 생성
        save_model(random_search, "random_search_model.pkl")
        save_model(le, "label_encoder.pkl")
    
        # 파일 불러오기
        df = pd.read_csv(datafile, encoding='cp949')

        # 데이터 전처리
        test = df[df.isna()[target]]
        train = df[df.notnull()[target]].copy()

        # 등급 숫자로 변경 및 라벨링
        le = LabelEncoder()
        train[target] = le.fit_transform(train[target])

        X_train = train.drop([target], axis=1)
        y_train = train[target]

        X_train = X_train[features]

        # ADASYN 객체
        sampling_strategy = {0: 1700, 3: 1700, 2: 1200, 4: 1200, 1: 700, 5: 700}
        adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
        X, y = adasyn.fit_resample(X_train, y_train)

        #모델 학습을 위해 train, test 분리
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        rf_classifier = RandomForestClassifier()

        param_dist = {
        'n_estimators': np.arange(50, 200, 10),
        'max_depth': [None] + list(np.arange(5, 30, 5)),
        'min_samples_split': np.arange(2, 11),
        'min_samples_leaf': np.arange(1, 11),
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
        }
        # 랜덤 서치 객체 생성
        random_search = RandomizedSearchCV(
            rf_classifier,
            param_distributions=param_dist,
            n_iter=100,  # 총 시도할 조합 횟수
            scoring='accuracy',  # 평가 지표
            cv=5,  # 교차 검증 폴드 수
            verbose=1,
            n_jobs=-1  # 모든 가용 CPU 코어 사용
        )

        # 랜덤 서치 수행
        random_search.fit(X_train, y_train)
    
        # 모델 저장
        save_model(random_search, "random_search_model.pkl")
        save_model(le, "label_encoder.pkl")

    predicted_grade = random_search.predict(input_data)
    predicted_grade_label = le.inverse_transform(predicted_grade)

    return predicted_grade_label, random_search, le


#사회 예측모델 
def social_process_data_and_predict(input_data, datafile, features, target):
     
     # PKL 파일 확인
    if os.path.exists("social_model.pkl") and os.path.exists("social_encoder.pkl"):
        random_search = load_model("social_model.pkl")
        le = load_model("social_encoder.pkl")
    else:
        random_search.fit(X_train, y_train)

        # pkl 파일 생성
        save_model(random_search, "social_model.pkl")
        save_model(le, "social_encoder.pkl")
        
        # 파일 불러오기
        df = pd.read_csv(datafile, encoding='cp949')

        # 데이터 전처리
        test = df[df.isna()[target]]
        train = df[df.notnull()[target]]

        # "D" 지우기
        train = train[train[target] != 'D']

        # 등급 숫자로 변경 및 라벨링
        le = LabelEncoder()
        train[target] = le.fit_transform(train[target])

        X_train = train.drop([target], axis=1)
        y_train = train[target]

        X_train = X_train[features]

        # ADASYN 객체    
        sampling_strategy = {0: 1700, 1: 1700, 2: 1200, 3: 1200, 4: 500}
        adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
        X, y = adasyn.fit_resample(X_train, y_train)
    
        # 모델 학습을 위해 train, test분리
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
        rf_classifier = RandomForestClassifier()

        param_dist = {
        'n_estimators': np.arange(50, 200, 10),
        'max_depth': [None] + list(np.arange(5, 30, 5)),
        'min_samples_split': np.arange(2, 11),
        'min_samples_leaf': np.arange(1, 11),
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
        }
        # 랜덤 서치 객체 생성
        random_search = RandomizedSearchCV(
            rf_classifier,
            param_distributions=param_dist,
            n_iter=100,  # 총 시도할 조합 횟수
            scoring='accuracy',  # 평가 지표
            cv=5,  # 교차 검증 폴드 수
            verbose=1,
            n_jobs=-1  # 모든 가용 CPU 코어 사용
        )

        # 랜덤 서치 수행
        random_search.fit(X_train, y_train)
    
        # 모델 저장
        save_model(random_search, "random_search_model.pkl")
        save_model(le, "label_encoder.pkl")

    predicted_grade = random_search.predict(input_data)
    predicted_grade_label = le.inverse_transform(predicted_grade)

    return predicted_grade_label, random_search, le

#지배구조 예측 모델
def gov_process_data_and_predict(input_data, datafile, features, target):
    
     # PKL 파일 확인
    if os.path.exists("random_search_model.pkl") and os.path.exists("label_encoder.pkl"):
        random_search = load_model("random_search_model.pkl")
        le = load_model("label_encoder.pkl")
    else:
        random_search.fit(X_train, y_train)
            
        # pkl 파일 생성
        save_model(random_search, "gov_model.pkl")
        save_model(le, "gov_encoder.pkl")
    
        # 파일 불러오기
        df = pd.read_csv(datafile, encoding='cp949')

        # 데이터 전처리
        test = df[df.isna()[target]]
        train = df[df.notnull()[target]].copy()
    
        # "D" 지우기
        train = train[train[target] != 'D']

        # 등급 숫자로 변경 및 라벨링
        le = LabelEncoder()
        train[target] = le.fit_transform(train[target])

        X_train = train.drop([target], axis=1)
        y_train = train[target]
        X_train = X_train[features]

        # ADASYN 객체 생성, 및 오버 샘플링
        sampling_strategy = {0: 1700, 3: 1700, 2: 1200, 1: 1200, 4: 500} # 1700 1700 1200 1200 500
        adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
        X, y = adasyn.fit_resample(X_train, y_train)
    
        # 오버샘플링 적용
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        rf_classifier = RandomForestClassifier()

        param_dist = {
        'n_estimators': np.arange(50, 200, 10),
        'max_depth': [None] + list(np.arange(5, 30, 5)),
        'min_samples_split': np.arange(2, 11),
        'min_samples_leaf': np.arange(1, 11),
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
        }
        # 랜덤 서치 객체 생성
        random_search = RandomizedSearchCV(
            rf_classifier,
            param_distributions=param_dist,
            n_iter=100,  # 총 시도할 조합 횟수
            scoring='accuracy',  # 평가 지표
            cv=5,  # 교차 검증 폴드 수
            verbose=1,
            n_jobs=-1  # 모든 가용 CPU 코어 사용
        )
 
        # 랜덤 서치 수행
        random_search.fit(X_train, y_train)
    
        # 모델 저장
        save_model(random_search, "gov_model.pkl")
        save_model(le, "gov_encoder.pkl")

    predicted_grade = random_search.predict(input_data)
    predicted_grade_label = le.inverse_transform(predicted_grade)

    return predicted_grade_label, random_search, le

#시나리오 모델
def scenario_analysis(model, encoder, data, scenario, features):
    # 새로운 시나리오에 따라 데이터를 조정
    modified_data = np.array(data) # 기존 데이터를 복사합니다.
    for feature in features:
        # 특성의 인덱스를 찾습니다.
        index = features.index(feature)
        # 해당 특성의 값을 변경합니다.
        modified_data[0][index] += scenario.get(feature, 0)

    # 수정된 데이터로 환경 등급 예측
    predicted_grade = model.predict(modified_data)

    # 예측된 환경 등급을 실제 환경 등급으로 디코딩
    predicted_grade_label = encoder.inverse_transform(predicted_grade)

    return predicted_grade_label[0]

#결과 출력
def create_report(model, encoder, features, data, scenario):
    # 원래 데이터에 대한 예측
    original_grade = model.predict(data)
    original_grade_label = encoder.inverse_transform(original_grade)

    # 시나리오에 따른 예측
    modified_grade_label = scenario_analysis(model, encoder, data, scenario, features)

    # 보고서 작성
    report = f"""
    Original Environmental Grade: {original_grade_label[0]} 
    
    Predicted Environmental Grade under scenario: {modified_grade_label} \n

    Scenario:
    """
    for feature, change in scenario.items():
        report += f"{feature}: {'increased' if change > 0 else 'decreased'} by {abs(change)} \n "
    report = report.replace('\n', '<br>')

    # 각 등급의 예측 확률을 시각화
    original_probs = model.predict_proba(data)

    # 시나리오에 따라 데이터를 조정
    modified_data = np.array(data) # 기존 데이터를 복사합니다.
    for feature in features:
        # 특성의 인덱스를 찾습니다.
        index = features.index(feature)
        # 해당 특성의 값을 변경합니다.
        modified_data[0][index] += scenario.get(feature, 0)

    modified_probs = model.predict_proba(modified_data)

    labels = encoder.classes_

    x = np.arange(len(labels))

    fig, ax = plt.subplots()

    bar_width = 0.35

    original_rects = ax.bar(x - bar_width/2, original_probs[0], bar_width, label='Original')
    modified_rects = ax.bar(x + bar_width/2, modified_probs[0], bar_width, label='Modified')

    ax.set_ylabel('Probabilities')
    ax.set_title('Probabilities by environmental grade and scenario')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.savefig('./static/graph.png')

    img = io.BytesIO()
    plt.savefig('graph.png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return report, plot_url

#메인 홈페이지
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

#환경 홈페이지
@app.route('/envi', methods=['GET', 'POST'])
def envi():
    if request.method == 'POST':
        # Get input feature values from the form

        greenhouse_gas = int(request.form['greenhouse gas'])
        energy_usage = int(request.form['energy usage'])
        Hazardous_Chemical = int(request.form['Hazardous Chemical'])
        water_usage = int(request.form['water usage'])
        waste_emissions = int(request.form['waste emissions'])

        # Create a 2D array of the input feature values
        input_data = [[greenhouse_gas, energy_usage, Hazardous_Chemical, water_usage, waste_emissions]]
        features = ["온실가스 배출량", "에너지 사용량", "유해화학물질 배출량", "용수 사용량", "폐기물 배출량"]
        datafile = './envi data.csv'
        target = "환경"

        # Process the data and make predictions
        predicted_grade = envi_process_data_and_predict(input_data, datafile, features, target)

        return render_template('envi.html', predicted_grade=predicted_grade[0])
    
    return render_template('envi.html', predicted_grade='')

#환경 시나리오
@app.route('/envi_scenario', methods=['GET', 'POST'])
def envi_scenario():

    if request.method == 'POST':
        # Get input feature values from the form

        greenhouse_gas = int(request.form['greenhouse gas'])
        energy_usage = int(request.form['energy usage'])
        Hazardous_Chemical = int(request.form['Hazardous Chemical'])
        water_usage = int(request.form['water usage'])
        waste_emissions = int(request.form['waste emissions'])

        # Create a 2D array of the input feature values
        input_data = [[greenhouse_gas, energy_usage, Hazardous_Chemical, water_usage, waste_emissions]]
        features = ["온실가스 배출량", "에너지 사용량", "유해화학물질 배출량", "용수 사용량", "폐기물 배출량"]
        datafile = './envi data.csv'
        target = "환경"

        scenario_greenhouse_gas = int(request.form['scenario greenhouse gas'])
        scenario_energy_usage = int(request.form['scenario energy usage'])
        scenario_Hazardous_Chemical = int(request.form['scenario Hazardous Chemical'])
        scenario_water_usage = int(request.form['scenario water usage'])
        scenario_waste_emissions = int(request.form['scenario waste emissions'])
        
        scenario = {
            "온실가스 배출량": scenario_greenhouse_gas,
            "에너지 사용량": scenario_energy_usage,
            "유해화학물질 배출량": scenario_Hazardous_Chemical,
            "용수 사용량": scenario_water_usage,
            "폐기물 배출량": scenario_waste_emissions,
            
        }

        # Process the data and make predictions
        predicted_grade, gb_clf, le = envi_process_data_and_predict(input_data, datafile, features, target)
        report, plot_url = create_report(gb_clf, le, features, input_data, scenario)

        return render_template('envi_scenario.html', predicted_grade=predicted_grade[0], report=report, plot_url=plot_url)
    return render_template('envi_scenario.html', predicted_grade='', report='', plot_url='')

#사회 홈페이지
@app.route('/social', methods=['GET', 'POST'])
def social():
    if request.method == 'POST':
        # Get input feature values from the form

        new_recruitment = float(request.form['New Recruitment'])
        resignation_retirement = float(request.form['resignation retirement'])
        female_workers = float(request.form['female workers'])
        training_hours = float(request.form['training hours'])
        social_contribution = int(request.form['social contribution'])
        industrial_accident = float(request.form['industrial accident'])

        # Create a 2D array of the input feature values
        input_data = [[new_recruitment, resignation_retirement, female_workers, training_hours, social_contribution, industrial_accident]]
        features = ["신규채용", "이직 및 퇴직", "여성 근로자 (합)", "교육 시간", "사회 공헌 및 투자", "산업재해"]
        datafile = './social data.csv'
        target = "사회"

        # Process the data and make predictions
        predicted_grade = social_process_data_and_predict(input_data, datafile, features, target)

        return render_template('social.html', predicted_grade=predicted_grade[0])
    
    return render_template('social.html', predicted_grade='')

#사회 시나리오
@app.route('/social_scenario', methods=['GET', 'POST'])
def social_scenario():

    if request.method == 'POST':
        # Get input feature values from the form

        new_recruitment = float(request.form['New Recruitment'])
        resignation_retirement = float(request.form['resignation retirement'])
        female_workers = float(request.form['female workers'])
        training_hours = float(request.form['training hours'])
        social_contribution = int(request.form['social contribution'])
        industrial_accident = float(request.form['industrial accident'])

        # Create a 2D array of the input feature values
        input_data = [[new_recruitment, resignation_retirement, female_workers, training_hours, social_contribution, industrial_accident]]
        features = ["신규채용", "이직 및 퇴직", "여성 근로자 (합)", "교육 시간", "사회 공헌 및 투자", "산업재해"]
        datafile = './social data.csv'
        target = "사회"

        scenario_new_recruitment = float(request.form['scenario New Recruitment'])
        scenario_resignation_retirement = float(request.form['scenario resignation retirement'])
        scenario_female_workers = float(request.form['scenario female workers'])
        scenario_training_hours = float(request.form['scenario training hours'])
        scenario_social_contribution = int(request.form['scenario social contribution'])
        scenario_industrial_accident = float(request.form['scenario industrial accident'])
        
        scenario = {
            "신규채용": scenario_new_recruitment,
            "이직 및 퇴직": scenario_resignation_retirement,
            "여성 근로자 (합)": scenario_female_workers,
            "교육 시간": scenario_training_hours,
            "사회 공헌 및 투자": scenario_social_contribution,
            "산업재해": scenario_industrial_accident

            
        }

        # Process the data and make predictions
        predicted_grade, gb_clf, le = social_process_data_and_predict(input_data, datafile, features, target)
        report, plot_url = create_report(gb_clf, le, features, input_data, scenario)

        return render_template('social_scenario.html', predicted_grade=predicted_grade[0], report=report, plot_url=plot_url)
    return render_template('social_scenario.html', predicted_grade='', report='', plot_url='')

#지배구조 홈페이지
@app.route('/gov', methods=['GET', 'POST'])
def gov():
    if request.method == 'POST':
        # Get input feature values from the form

        attendance_rate = float(request.form['attendance rate'])
        board_of_directors = int(request.form['board of directors'])
        number_of_board_members = int(request.form['number of board members'])
        audit_committee = int(request.form['audit committee'])
        stock = float(request.form['stock'])

        # Create a 2D array of the input feature values
        input_data = [[attendance_rate, board_of_directors, number_of_board_members, audit_committee, stock]]
        features = [ "이사 출석률","이사회 현황 횟수","이사 명수","감사위원회 운영 횟수","발행 주식수 / 발행 가능 주식수"]
        datafile = './gov data.csv'
        target = "지배구조"

        # Process the data and make predictions
        predicted_grade = gov_process_data_and_predict(input_data, datafile, features, target)

        return render_template('gov.html', predicted_grade=predicted_grade[0])
    
    return render_template('gov.html', predicted_grade='')

#지배구조 시나리오
@app.route('/gov_scenario', methods=['GET', 'POST'])
def gov_scenario():

    if request.method == 'POST':
        # Get input feature values from the form

        attendance_rate = float(request.form['attendance rate'])
        board_of_directors = int(request.form['board of directors'])
        number_of_board_members = int(request.form['number of board members'])
        audit_committee = int(request.form['audit committee'])
        stock = float(request.form['stock'])

        # Create a 2D array of the input feature values
        input_data = [[attendance_rate, board_of_directors, number_of_board_members, audit_committee, stock]]
        features = [ "이사 출석률","이사회 현황 횟수","이사 명수","감사위원회 운영 횟수","발행 주식수 / 발행 가능 주식수"]
        datafile = './gov data.csv'
        target = "지배구조"

        scenario_attendance_rate = int(request.form['scenario attendance rate'])
        scenario_board_of_directors = int(request.form['scenario board of directors'])
        scenario_number_of_board_members = int(request.form['scenario number of board members'])
        scenario_audit_committee = int(request.form['scenario audit committee'])
        scenario_stock = int(request.form['scenario stock'])
        
        scenario = {
            "이사 출석률": scenario_attendance_rate,
            "이사회 현황 횟수": scenario_board_of_directors,
            "이사 명수": scenario_number_of_board_members,
            "감사위원회 운영 횟수": scenario_audit_committee,
            "발행 주식수/발행 가능 주식수": scenario_stock,
            
        }

        # Process the data and make predictions
        predicted_grade, gb_clf, le = envi_process_data_and_predict(input_data, datafile, features, target)
        report, plot_url = create_report(gb_clf, le, features, input_data, scenario)

        return render_template('gov_scenario.html', predicted_grade=predicted_grade[0], report=report, plot_url=plot_url)
    return render_template('gov_scenario.html', predicted_grade='', report='', plot_url='')

#호스트 지정 + 웹 페이지 배포
if __name__ == '__main__':
    app.run( port=80, debug= True, host='0.0.0.0')
