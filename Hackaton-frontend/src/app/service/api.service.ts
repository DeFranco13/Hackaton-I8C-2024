import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private BASE_URL = 'http://10.1.21.26:5000/'; // Adjust as needed

  constructor(private http: HttpClient) { }

  uploadFile(file: File) {
    const formData: FormData = new FormData();
    formData.append('file', file, file.name);
    return this.http.post(`${this.BASE_URL}/upload`, formData);
  }

  getData(): Observable<any>{
    return this.http.get(`${this.BASE_URL}/anomalies`);
  }
}
