import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

#,
app = Flask(__name__)

#환경 예측모델
def envi_process_data_and_predict(input_data, datafile, features, target):
    
    # 파일 불러오기
    df = pd.read_csv(datafile, encoding='cp949')

    # 데이터 전처리
    test = df[df.isna()[target]]
    train = df[df.notnull()[target]].copy()

    # 등급 숫자로 변경 및 라벨링
    label_mapping = {'D': 0,'C': 1, 'B': 2, 'B+': 3, 'A': 4, 'A+': 5}
    train[target] = train[target].map(label_mapping)

    X_train = train.drop([target], axis=1)
    y_train = train[target]

    X_train = X_train[features]

    # ADASYN 객체
    sampling_strategy = {0: 1700, 3: 1700, 2: 1200, 4: 1200, 1: 700, 5: 700}
    adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

    # Gradient Boosting Classifier 생성
    gb_clf = GradientBoostingClassifier(random_state=42)

    # 모델 훈련
    gb_clf.fit(X_train_adasyn, y_train_adasyn)

    # Predict the grade with the entered feature values
    predicted_grade = gb_clf.predict(input_data)

    return predicted_grade

#사회 예측모델 
def social_process_data_and_predict(input_data, datafile, features, target):
    # 파일 불러오기
    df = pd.read_csv(datafile, encoding='cp949')

    # 데이터 전처리
    test = df[df.isna()[target]]
    train = df[df.notnull()[target]]

    # "D" 지우기
    train = train[train[target] != 'D']

    # 등급 숫자로 변경 및 라벨링
    label_mapping = {'C': 0, 'B': 1, 'B+': 2, 'A': 3, 'A+': 4}
    train[target] = train[target].map(label_mapping)

    X_train = train.drop([target], axis=1)
    y_train = train[target]

    X_train = X_train[features]

    # ADASYN 객체
    sampling_strategy = {0: 1700, 1: 1700, 2: 1200, 3: 1200, 4: 500}
    adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
    
    # Gradient Boosting Classifier 생성
    gb_clf = GradientBoostingClassifier(random_state=42)

    # 모델 훈련
    gb_clf.fit(X_train_adasyn, y_train_adasyn)

    # Predict the grade with the entered feature values
    predicted_grade = gb_clf.predict(input_data)

    return predicted_grade

#지배구조 예측 모델
def gov_process_data_and_predict(input_data, datafile, features, target):
    
    # 파일 불러오기
    df = pd.read_csv(datafile, encoding='cp949')

    # 데이터 전처리
    test = df[df.isna()[target]]
    train = df[df.notnull()[target]].copy()
    
       # "D" 지우기
    train = train[train[target] != 'D']

    # 등급 숫자로 변경 및 라벨링
    label_mapping = {'C': 0, 'B': 1, 'B+': 2, 'A': 3, 'A+': 4}
    train[target] = train[target].map(label_mapping)

    X_train = train.drop([target], axis=1)
    y_train = train[target]

    X_train = X_train[features]

    # ADASYN 객체 생성
    adasyn = ADASYN(sampling_strategy='minority', random_state=42)

    # 오버샘플링 적용
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

    sampling_strategy = {0: 1700, 3: 1700, 2: 1200, 1: 1200, 4: 500} # 1700 1700 1200 1200 500
    adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

    # Gradient Boosting Classifier 생성
    gb_clf = GradientBoostingClassifier(random_state=42)

    # 모델 훈련
    gb_clf.fit(X_train_adasyn, y_train_adasyn)

    # Predict the grade with the entered feature values
    predicted_grade = gb_clf.predict(input_data)

    return predicted_grade


#메인 홈페이지
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

#환경
@app.route('/envi', methods=['GET', 'POST'])
def envi():
    if request.method == 'POST':
        # Get input feature values from the form

        greenhouse_gas = float(request.form['greenhouse gas'])
        energy_usage = float(request.form['energy usage'])
        Hazardous_Chemical = float(request.form['Hazardous Chemical'])
        water_usage = int(request.form['water usage'])
        waste_emissions = int(request.form['waste emissions'])

        # Create a 2D array of the input feature values
        input_data = [[greenhouse_gas, energy_usage, Hazardous_Chemical, water_usage, waste_emissions]]
        features = ["온실가스 배출량", "에너지 사용량", "유해화학물질 배출량", "용수 사용량", "폐기물 배출량"]
        datafile = './envi data.csv'
        target = "환경"

        # Process the data and make predictions
        predicted_grade = envi_process_data_and_predict(input_data, datafile, features, target)

        # Map the predicted grade to the corresponding social grade label
        label_mapping_reverse = {0: 'D', 1: 'C', 2: 'B', 3: 'B+', 4: 'A', 5: 'A+'}
        predicted_grade_label = label_mapping_reverse[predicted_grade[0]]

        return render_template('envi.html', predicted_grade=predicted_grade_label)
    
    return render_template('envi.html', predicted_grade='')

#사회
@app.route('/social', methods=['GET', 'POST'])
def social():
    if request.method == 'POST':
        # Get input feature values from the form

        new_recruitment = float(request.form['New Recruitment'])
        resignation_retirement = float(request.form['resignation_retirement'])
        female_workers = float(request.form['female_workers'])
        training_hours = int(request.form['training_hours'])
        social_contribution = int(request.form['social_contribution'])
        industrial_accident = float(request.form['industrial_accident'])

        # Create a 2D array of the input feature values
        input_data = [[new_recruitment, resignation_retirement, female_workers, training_hours, social_contribution, industrial_accident]]
        features = ["신규채용", "이직 및 퇴직", "여성 근로자 (합)", "교육 시간", "사회 공헌 및 투자", "산업재해"]
        datafile = './social data.csv'
        target = "사회"

        # Process the data and make predictions
        predicted_grade = social_process_data_and_predict(input_data, datafile, features, target)

        # Map the predicted grade to the corresponding social grade label
        label_mapping_reverse = {0: 'C', 1: 'B', 2: 'B+', 3: 'A', 4: 'A+'}
        predicted_grade_label = label_mapping_reverse[predicted_grade[0]]

        return render_template('social.html', predicted_grade=predicted_grade_label)
    
    return render_template('social.html', predicted_grade='')

#지배구조
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

        # Map the predicted grade to the corresponding social grade label
        label_mapping_reverse = {0: 'C', 1: 'B', 2: 'B+', 3: 'A', 4: 'A+'}
        predicted_grade_label = label_mapping_reverse[predicted_grade[0]]

        return render_template('gov.html', predicted_grade=predicted_grade_label)
    
    return render_template('gov.html', predicted_grade='')

#호스트 지정 + 웹 페이지 배포
if __name__ == '__main__':
    app.run(port = 80, debug=True)
