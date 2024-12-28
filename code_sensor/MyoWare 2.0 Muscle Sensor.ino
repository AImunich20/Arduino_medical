void setup() 
{
  Serial.begin(115200);
  Serial.println("MyoWare analogRead_SINGLE");
}
void loop() 
{  
  int sensorValue = analogRead(A0); 
  Serial.println(sensorValue);
  delay(50);
}
