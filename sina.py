import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, redirect, url_for
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import io
import base64


app = Flask(__name__)

#환경 예측모델
def envi_process_data_and_predict(input_data, datafile, features, target):
    
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Gradient Boosting Classifier 생성
    gb_clf = GradientBoostingClassifier(n_estimators=230, learning_rate=0.15, max_depth=4, min_samples_leaf=40, ccp_alpha=0.00000015, random_state=42)

    # 모델 훈련
    gb_clf.fit(X_train, y_train)

    # 예측값 도출
    predicted_grade = gb_clf.predict(input_data)
    predicted_grade_label = le.inverse_transform(predicted_grade)


    return predicted_grade_label, gb_clf, le

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

def create_report(model, encoder, features, data, scenario):
    # 원래 데이터에 대한 예측
    original_grade = model.predict(data)
    original_grade_label = encoder.inverse_transform(original_grade)

    # 시나리오에 따른 예측
    modified_grade_label = scenario_analysis(model, encoder, data, scenario, features)

    # 보고서 작성
    report = f"""
    Original Environmental Grade: {original_grade_label[0]}
    Predicted Environmental Grade under scenario: {modified_grade_label}

    Scenario:
    """
    for feature, change in scenario.items():
        report += f"{feature}: {'increased' if change > 0 else 'decreased'} by {abs(change)}\n"

    print(report)

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

        # Convert the plot to a string encoded as base64

    plt.show()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return report, plot_url


#메인 홈페이지
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

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

#호스트 지정 + 웹 페이지 배포
if __name__ == '__main__':
    app.run( port=80, debug= True, host='0.0.0.0')